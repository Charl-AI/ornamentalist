"""Using the ornamentalist CLI generation feature."""

import ornamentalist
from ornamentalist import Configurable


@ornamentalist.configure(cli_defaults={"a": 5})
def multiply(a: int = Configurable, b: int = Configurable):
    print(a * b)


if __name__ == "__main__":
    configs = ornamentalist.cli()
    print(configs)

    # the current setup allows you to run sweeps by specifying multiple args
    # if you only ever want to run one job at a time, you can do:
    #     assert len(configs) == 0
    #     ornamentalist.setup(configs[0])
    #     multiply()

    for config in configs:
        ornamentalist.setup(config, force=True)
        multiply()

# Examples:

# --- 1. Single run:
# > python examples/cli.py --multiply.a  2 --multiply.b  4
# > [Namespace(**{'multiply.a': 2, 'multiply.b': 4})]
# > 8

# --- 2. Sweep using default value for a:
# > python examples/cli.py --multiply.b 2 3
# > [Namespace(**{'multiply.a': 5, 'multiply.b': 2}), Namespace(**{'multiply.a': 5, 'multiply.b': 3})]
# > 10
# > 15

# --- 3. Sweep over configured values of a and b:
# > python examples/cli.py --multiply.a 1 2 --multiply.b 2 3
# > [Namespace(**{'multiply.a': 1, 'multiply.b': 2}), Namespace(**{'multiply.a': 1, 'multiply.b': 3}), Namespace(**{'multiply.a': 2, 'multiply.b': 2}), Namespace(**{'multiply.a': 2, 'multiply.b': 3})]
# > 2
# > 3
# > 4
# > 6

# --- 4. Auto-generated help message:
# python examples/cli.py --help
# usage: cli.py [-h] [--multiply.a  ...]] --multiply.b  ...]
#
# options:
#   -h, --help            show this help message and exit
#
# multiply:
#   Hyperparameters for __main__.multiply
#
#   --multiply.a  ...]
#                         Type: int (optional), default=5
#   --multiply.b  ...]
#                         Type: int (required)
