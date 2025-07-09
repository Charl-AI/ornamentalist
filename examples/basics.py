"""Basic usage of ornamentalist."""

import ornamentalist
from ornamentalist import Configurable


# setting verbose=True is useful for debugging
@ornamentalist.configure(verbose=True)
def add_n(x: int, n: int = Configurable):
    print(x + n)


# by default, ornamentalist looks for parameters
# in CONFIG_DICT[func.__name__],
# you can override this with a custom key like so
@ornamentalist.configure(name="greeting_config")
def greet(name: str = Configurable):
    print(f"Hello, {name}")


# you can even use ornamentalist on classes!
class MyClass:
    # you probably want to give constructors custom
    # names, else they will just be "__init__"
    @ornamentalist.configure(name="myclass.init")
    def __init__(self, a: float = Configurable):
        print(a)


if __name__ == "__main__":
    # you can manually supply config with argparse, hydra etc.
    # we also provide ornamentalist.cli() to automatically
    # generate a basic CLI.
    # But we will hardcode it for this example...
    config = {
        "add_n": {"n": 5},
        # greeting_config and myclass_init are the
        # custom names we specified earlier
        "greeting_config": {"name": "Alice"},
        "myclass.init": {"a": 4.5},
    }
    ornamentalist.setup(config)

    add_n(10)
    greet()
    MyClass()

    # you can access the config dict anywhere in your program
    # through `ornamentalist.get_config()`
    assert ornamentalist.get_config() == config

# Output:
# > python examples/basics.py
# > INFO:ornamentalist:Injecting parameters {'n': 5} into __main__.add_n
# > 15
# > Hello, Alice
# > 4.5
