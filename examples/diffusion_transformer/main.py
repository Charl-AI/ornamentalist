import argparse
import functools
import logging
import os
import pathlib
import time

import submitit
import torch
from submitit.helpers import RsyncSnapshot, clean_env
from torch.nn.parallel import DistributedDataParallel as DDP

import ornamentalist
from examples.diffusion_transformer.distributed import Distributed
from examples.diffusion_transformer.dit import get_model_cls
from examples.diffusion_transformer.mnist import get_dataloaders
from examples.diffusion_transformer.trainer import TrainState, train

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(config):
    # call ornamentalist.setup() here since submitit may launch this
    # in a different process to where you called launch()
    ornamentalist.setup(config, force=True)

    D = Distributed()  # setup distributed environment

    # update output_dir (to prevent overwriting outputs in array jobs/sweeps)
    job_env = submitit.JobEnvironment()
    print(f"Running job ID {job_env.job_id}")
    output_dir = config["launcher"]["output_dir"]
    output_dir = os.path.join(output_dir, f"{job_env.job_id}")

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
        output_dir=output_dir,
        D=D,
    )
    del D  # cleanup distributed environment (just to be safe)


def launch():
    """Thin wrapper that launches the main function with submitit."""

    parser = argparse.ArgumentParser(formatter_class=argparse.MetavarTypeHelpFormatter)
    group = parser.add_argument_group("launcher")
    group.add_argument("--launcher.nodes", type=int, default=1)
    group.add_argument("--launcher.gpus", type=int, default=1, help="(per node)")
    group.add_argument("--launcher.cpus", type=int, default=16, help="(per GPU)")
    group.add_argument("--launcher.ram", type=int, default=32, help="(GiB per GPU)")
    group.add_argument("--launcher.timeout", type=int, default=60, help="(minutes)")
    group.add_argument("--launcher.partition", type=str, default="hopper")
    group.add_argument(
        "--launcher.cluster",
        type=str,
        choices=["debug", "local", "slurm"],
        default="debug",
    )
    group.add_argument(
        "--launcher.output_dir",
        type=str,
        default="/mnt/ps/home/CORP/charlie.jones/project/ornamentalist/outputs",
    )
    group.add_argument(
        "--launcher.qos",
        type=str,
        default="normal",
        choices=["normal", "high"],
    )

    # auto-generate the rest of the CLI
    configs = ornamentalist.cli(parser)

    # create run-specific output directory
    output_dir = os.path.join(
        configs[0]["launcher"]["output_dir"], f"{time.time():.0f}"
    )
    configs[0]["launcher"]["output_dir"] = output_dir
    cluster = configs[0]["launcher"]["cluster"]

    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)
    executor.update_parameters(
        # name=configs[0]["launcher"],
        slurm_partition=configs[0]["launcher"]["partition"],
        slurm_qos=configs[0]["launcher"]["qos"],
        nodes=configs[0]["launcher"]["nodes"],
        tasks_per_node=configs[0]["launcher"]["gpus"],  # set ntasks = ngpus
        gpus_per_node=configs[0]["launcher"]["gpus"],
        cpus_per_task=configs[0]["launcher"]["cpus"],
        slurm_mem_per_gpu=configs[0]["launcher"]["ram"],
        timeout_min=configs[0]["launcher"]["timeout"],
        stderr_to_stdout=True,
        # send kill signal 2 minutes before timeout to give time to save checkpoint
        slurm_signal_delay_s=120,
    )

    # it's good practice to use RsyncSnapshot to ensure that changes to your
    # code don't affect jobs in the queue
    snapshot_dir = os.path.join(output_dir, "snapshot")
    with RsyncSnapshot(pathlib.Path(snapshot_dir)), clean_env():
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
    launch()
