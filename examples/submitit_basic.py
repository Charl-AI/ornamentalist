import argparse
import functools
import time

import submitit

import ornamentalist


@ornamentalist.configure()
def multiply(a: int = ornamentalist.Configurable, b: int = ornamentalist.Configurable):
    print(a * b)


def main(config):
    # call ornamentalist.setup() here since submitit may launch this
    # in a different process to where you called launch()
    ornamentalist.setup(config, force=True)
    multiply()


def launch():
    """Thin wrapper that launches the main function with submitit."""

    # add some extra launcher params to the auto-generated parser
    # we use the same group.parameter pattern as the auto-generated one
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("launcher")
    group.add_argument("--launcher.output_dir", type=str, default="./outputs")
    group.add_argument(
        "--launcher.cluster", type=str, choices=["debug", "local"], default="debug"
    )

    configs = ornamentalist.cli(parser)

    output_dir = configs[0]["launcher"]["output_dir"] + f"/{time.time():.0f}"
    cluster = configs[0]["launcher"]["cluster"]

    executor = submitit.AutoExecutor(folder=output_dir, cluster=cluster)

    fns = [functools.partial(main, config=config) for config in configs]
    jobs = executor.submit_array(fns)
    _ = [j.results()[0] for j in jobs]


if __name__ == "__main__":
    launch()
