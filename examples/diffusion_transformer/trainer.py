import dataclasses
import logging
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm import tqdm

import ornamentalist
from examples.diffusion_transformer.distributed import Distributed, rank_zero

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


@rank_zero()
def log(*, step: int, msg: str):
    _log.info(f"[{step}] {msg}")


@dataclasses.dataclass
class TrainState:
    ddp: DDP
    opt: torch.optim.Optimizer
    global_step: int


@ornamentalist.configure()
def guided_prediction(
    model: torch.nn.Module,
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
    cfg_omega: float = ornamentalist.Configurable[1.0],
    cfg_alpha: float = ornamentalist.Configurable[0.0],
) -> torch.Tensor:  # fmt: off
    cond = model(t=t, x=x, y=y)
    null_label = model.y_embedder.num_classes  # type: ignore
    y_null = torch.full_like(y, null_label)  # type: ignore
    uncond = model(t=t, x=x, y=y_null)
    diff = cond - uncond
    scale = torch.norm(diff, p=2, dim=tuple(range(1, diff.ndim))) ** cfg_alpha
    scale = scale.reshape(-1, 1, 1, 1).expand_as(diff)
    guidance = cfg_omega * diff * scale
    vt = uncond + guidance
    return vt


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
@ornamentalist.configure()
def generate(
    model: torch.nn.Module,
    x0: torch.Tensor,  # shape (B, C, H, W), sampled from N(0, I)
    y1: torch.Tensor,  # shape (B,) - [0,num_classes) with last class reserved as null token
    show_progress: bool = False,
    n_steps: int = ornamentalist.Configurable[100],
) -> torch.Tensor:
    """Generate a sample with the Euler probability flow ODE solver."""
    t = torch.linspace(0, 1, n_steps).to(x0.device)
    dt = 1 / n_steps
    x1 = x0
    for i in tqdm(
        range(n_steps),
        desc="Generating batch of samples",
        disable=not show_progress,
    ):
        t_ = t[i].expand(x0.shape[0])
        x1 += guided_prediction(model=model, t=t_, x=x1, y=y1) * dt
    return x1


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
@ornamentalist.configure()
def compute_loss(
    model: torch.nn.Module,  # maps t,x,y -> velocity
    x0: torch.Tensor,        # shape (B, C, H, W), sampled from N(0, I)
    x1: torch.Tensor,        # shape (B, C, H, W), sampled from data distribution
    y1: torch.Tensor,        # shape (B,) - [0,num_classes) with last class reserved as null token
    sigma: float = ornamentalist.Configurable[0.0], # if 0, Dirac marginals, else Gaussian
) -> torch.Tensor:  # fmt: off
    """Compute the flow matching (linear interpolant) loss."""
    t = torch.rand_like(y1, dtype=torch.float32)  # shape (B,) - [0,1)
    t_ = t.reshape(-1, 1, 1, 1).expand_as(x0)  # B -> B,C,H,W

    mu = torch.lerp(x0, x1, t_)
    ut = x1 - x0

    eps = torch.randn_like(x0)
    xt = mu + eps * sigma

    ut_pred = model(t=t, x=xt, y=y1)
    return torch.nn.functional.mse_loss(ut_pred, ut)


@torch.inference_mode()
def evaluate(
    *,
    state: TrainState,
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    D: Distributed,
) -> None:
    model = state.ddp.module
    model.eval()

    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)

    for val_step, batch in enumerate(
        tqdm(
            loader,
            desc="Evaluating",
            total=len(loader),
            unit="step",
            disable=D.rank != 0,  # only show from one process
        )
    ):
        x1, y1 = batch
        x1 = x1.to(D.device, non_blocking=True)
        y1 = y1.to(D.device, non_blocking=True)

        x1 = x1.to(torch.float32) / 127.5 - 1
        x0 = torch.randn_like(x1)
        y1 = y1.to(torch.int64)

        loss = compute_loss(model=model, x0=x0, x1=x1, y1=y1)
        running_loss += loss * x1.shape[0]
        num_samples += x1.shape[0]

        if val_step == 0:  # generate and plot samples on first batch
            NUM_TO_PLOT = 16
            assert NUM_TO_PLOT <= x0.shape[0] * D.world_size
            preds = generate(model=model, x0=x0, y1=y1)
            preds = (preds + 1) / 2  # (-1, 1) -> (0, 1)
            preds = D.gather_concat(preds)  # B,C,H,W -> world_size * B,C,H,W
            idxs = torch.randperm(preds.shape[0])[:NUM_TO_PLOT]
            preds = preds[idxs]

            os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
            fp = os.path.join(output_dir, f"samples/grid_{state.global_step}.png")
            save_image(preds, fp, nrow=NUM_TO_PLOT // 4)

    running_loss = D.all_reduce(running_loss, op="sum")
    num_samples = D.all_reduce(num_samples, op="sum")

    val_loss = (running_loss / num_samples).item()
    log(step=state.global_step, msg=f"val_loss: {val_loss}")
    D.barrier()
    return


def infinite_dataloader(dataloader: torch.utils.data.DataLoader):
    # assumes you are using a distributed sampler with set_epoch
    sampler = getattr(dataloader, "sampler", None)
    current_epoch = 0
    while True:  # loop indefinitely over epochs
        sampler.set_epoch(current_epoch)  # type: ignore
        epoch_iterator = iter(dataloader)
        yield from epoch_iterator
        current_epoch += 1


@ornamentalist.configure()
def train(
    *,
    state: TrainState,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    output_dir: str,
    D: Distributed,
    num_steps: int = ornamentalist.Configurable[10_000],
    log_every_n_steps: int = ornamentalist.Configurable[100],
    eval_every_n_steps: int = ornamentalist.Configurable[1000],
) -> None:
    running_loss = torch.tensor(0.0, device=D.device)
    num_samples = torch.tensor(0, device=D.device)
    start_time = time.time()

    loader = infinite_dataloader(train_loader)
    for _, batch in tqdm(
        zip(range(state.global_step, num_steps), loader),
        desc="Training",
        initial=state.global_step,
        total=num_steps,
        unit="step",
        disable=D.rank != 0,  # only show from one process
    ):
        state.ddp.train()
        x1, y1 = batch
        x1 = x1.to(D.device, non_blocking=True)
        y1 = y1.to(D.device, non_blocking=True)

        x1 = x1.to(torch.float32) / 127.5 - 1
        x1 = x1 + torch.rand_like(x1) / 127.5  # uniform dequantization
        x0 = torch.randn_like(x1)
        y1 = y1.to(torch.int64)

        loss = compute_loss(model=state.ddp, x0=x0, x1=x1, y1=y1)
        running_loss += loss.detach() * x0.shape[0]  # total batch loss
        num_samples += x0.shape[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.ddp.parameters(), max_norm=1.0)

        state.opt.step()
        state.opt.zero_grad()
        state.global_step += 1

        if state.global_step % log_every_n_steps == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            throughput_per_gpu = num_samples / elapsed_time

            running_loss = D.all_reduce(running_loss, op="sum")
            num_samples = D.all_reduce(num_samples, op="sum")

            throughput_total = num_samples / elapsed_time
            loss = (running_loss / num_samples).item()
            running_loss = torch.tensor(0.0, device=D.device)
            num_samples = torch.tensor(0, device=D.device)

            log(step=state.global_step, msg=f"{loss=}")
            log(step=state.global_step, msg=f"{throughput_total=}")
            log(step=state.global_step, msg=f"{throughput_per_gpu=}")
            log(step=state.global_step, msg=f"{grad_norm=}")

        if state.global_step % eval_every_n_steps == 0:
            evaluate(
                state=state,
                loader=val_loader,
                output_dir=output_dir,
                D=D,
            )

    log(step=state.global_step, msg="Training complete")
    return
