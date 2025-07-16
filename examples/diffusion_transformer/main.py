import functools
import logging
import os
import pathlib
import time
from typing import Literal

import submitit
import torch
from submitit.helpers import RsyncSnapshot
from torch.nn.parallel import DistributedDataParallel as DDP

import ornamentalist
from examples.diffusion_transformer.distributed import Distributed
from examples.diffusion_transformer.dit import get_model_cls
from examples.diffusion_transformer.mnist import get_dataloaders
from examples.diffusion_transformer.trainer import TrainState, train

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(config):
    ornamentalist.setup(config, force=True)
    with Distributed() as D:
        job_id = submitit.JobEnvironment().job_id
        output_dir = pathlib.Path.cwd().parent / job_id
        log.info(f"Running job ID {job_id}")
        log.info(f"Using {output_dir} as output directory")

        model_cls = get_model_cls()
        net = model_cls(input_size=28, in_channels=1, num_classes=10, learn_sigma=False)
        net.to(D.device)
        ddp = DDP(net)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        state = TrainState(ddp=ddp, opt=opt, global_step=0)

        train_loader, val_loader = get_dataloaders()
        train(
            state=state,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(output_dir),
            D=D,
        )


@ornamentalist.configure()
def launcher(
    configs: list[ornamentalist.ConfigDict],
    nodes: int = ornamentalist.Configurable[1],
    gpus: int = ornamentalist.Configurable[1],
    cpus: int = ornamentalist.Configurable[16],
    ram: int = ornamentalist.Configurable[64],
    timeout: int = ornamentalist.Configurable[20],
    partition: str = ornamentalist.Configurable["hopper"],
    qos: str = ornamentalist.Configurable["normal"],
    output_dir: str = ornamentalist.Configurable["./outputs/"],
    cluster: Literal["debug", "local", "slurm"] = ornamentalist.Configurable["debug"],
):
    """Thin wrapper that launches the main function with submitit."""

    # create run-specific output directory
    output_dir = os.path.join(output_dir, f"{time.time():.0f}")

    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_qos=qos,
        nodes=nodes,
        tasks_per_node=gpus,  # set ntasks = ngpus
        gpus_per_node=gpus,
        cpus_per_task=cpus,
        slurm_mem_per_gpu=f"{ram}G",
        timeout_min=timeout,
        stderr_to_stdout=True,
        slurm_signal_delay_s=120,
    )

    # it's good practice to use RsyncSnapshot to ensure that changes to your
    # code don't affect jobs in the queue
    snapshot_dir = os.path.join(output_dir, "snapshot")
    with RsyncSnapshot(pathlib.Path(snapshot_dir)):
        fns = [functools.partial(main, config=config) for config in configs]
        jobs = executor.submit_array(fns)
        log.info(f"Submitted {jobs=}")

        # if local or debug, wait for job to finish, otherwise exit script as soon as job is submitted
        if cluster == "local":
            log.info("Running job(s) locally using multiprocessing...")
            log.info(f"stdout and stderr for each process are logged to {output_dir}.")
            log.info("The job is in another process so you won't see anything here.")
            log.info("(But ctrl-c will still kill the job.)")
            _ = [j.results()[0] for j in jobs]

        elif cluster == "debug":
            log.info("Running job(s) in this process in debug mode...")
            log.info("pdb will open automatically on crash.")
            log.info("It's best to only use 1 GPU in this mode.")
            _ = [j.results()[0] for j in jobs]


if __name__ == "__main__":
    configs = ornamentalist.cli()
    # if launching a sweep, all configs must have same launcher settings
    assert all(config["launcher"] == configs[0]["launcher"] for config in configs)
    ornamentalist.setup(configs[0], force=True)
    launcher(configs=configs)
