"""Using the ornamentalist CLI generation feature."""

from typing import Literal

import ornamentalist
from ornamentalist import Configurable


@ornamentalist.configure()
def multiply(
    a: int = Configurable[5],  # 5 is the default value for the CLI, so a is optional
    b: Literal[2, 3, 4] = Configurable,  # no default value given, so b is required
    # c is optional, you can pass an int or null/NULL/None/none on the CLI
    c: int | None = Configurable[None],
):
    if c is None:
        print(a * b)
    else:
        print(a * b * c)


if __name__ == "__main__":
    configs = ornamentalist.cli()
    print(configs)

    # the current setup allows you to run sweeps by specifying multiple args
    # if you only ever want to run one job at a time, you can do:
    #     assert len(configs) == 1
    #     ornamentalist.setup(configs[0])
    #     multiply()

    for config in configs:
        ornamentalist.setup(config, force=True)
        multiply()

# Examples:

# --- 1. Single run:

# > python examples/cli.py --multiply.a  2 --multiply.b  4
# > [{'multiply': {'a': 2, 'b': 4, 'c': None}}]
# > 8

# --- 2. Sweep over a={5} x b={2,3} x c={None} (2 runs total)

# > python examples/cli.py --multiply.b 2 3
# > [{'multiply': {'a': 5, 'b': 2, 'c': None}}, {'multiply': {'a': 5, 'b': 3, 'c': None}}]
# > 10
# > 15

# --- 3. Sweep over a={1,2} x b={2,3} x c={5,None} (8 runs total)

# > python examples/cli.py --multiply.a 1 2 --multiply.b 2 3 --multiply.c 5 none
# > [{'multiply': {'a': 1, 'b': 2, 'c': 5}}, {'multiply': {'a': 1, 'b': 2, 'c': None}},
#    {'multiply': {'a': 1, 'b': 3, 'c': 5}}, {'multiply': {'a': 1, 'b': 3, 'c': None}},
#    {'multiply': {'a': 2, 'b': 2, 'c': 5}}, {'multiply': {'a': 2, 'b': 2, 'c': None}},
#    {'multiply': {'a': 2, 'b': 3, 'c': 5}}, {'multiply': {'a': 2, 'b': 3, 'c': None}}]
# > 10
# > 2
# > 15
# > 3
# > 20
# > 4
# > 30
# > 6

# --- 4. Auto-generated help message:

# python examples/cli.py --help
# usage: cli.py [-h] [--multiply.a  ...]] --multiply.b  ...] [--multiply.c  ...]]
#
# options:
#   -h, --help            show this help message and exit
#
# multiply:
#   Hyperparameters for __main__.multiply
#
#   --multiply.a  ...] Type: int (optional), default=5
#   --multiply.b  ...] Type: int, choices: (2, 3, 4) (required)
#   --multiply.c  ...] Type: int | None (optional), default=None
