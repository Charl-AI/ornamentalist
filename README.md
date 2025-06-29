# Ornamentalist

Ornamentalist is a tiny library for configuring functions with fixed hyperparameters in Python. The goal is to allow research code to be more flexible and hackable, without losing readability.

The core thing ornamentalist does is it allows you to specify the parameters of a function as `Configurable`. Ornamentalist will then replace the function with a 'partial' version of itself, where the parameters are fixed to the values supplied by your configuration.

I encourage you to read the short [blog post](https://charl-ai.github.io/blog/args) to understand the motivation behind this libary and why I think it's a good solution. For worked examples of how to use ornamentalist with other tools such as hydra, argparse, or submitit, check out the `examples/` directory.

You can install ornamentalist with pip:

```
pip install ornamentalist
```

Ornamentalist is only 1-file, so feel free to copy-paste it into your projects if you prefer.

## Usage

Using ornamentalist is straightforward:

1. Mark hyperparameters as configurable by setting their default value to `ornamentalist.Configurable`.
2. Decorate the function with `@ornamentalist.configure()`.
3. Create a config dictionary at the start of your program (usually, these values will be suppled by argparse or hydra etc.)
4. Call `ornamentalist.setup_config(config)` before running any configurable functions.

## Quickstart

Tip: You can find this file in `examples/basics.py`. Download and play with it to get a feel for ornamentalist :).

```python

import ornamentalist
from ornamentalist import Configurable


# basic usage of ornamentalist...
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
    # usually config would be supplied by argparse or
    # hydra etc. but we will hardcode it here...
    config = {
        "add_n": {"n": 5},
        # greeting_config and myclass_init are the
        # custom names we specified earlier
        "greeting_config": {"name": "Alice"},
        "myclass.init": {"a": 4.5},
    }
    ornamentalist.setup_config(config)

    add_n(10)
    greet()
    c = MyClass()

    # you can access the config dict anywhere in your program
    # through `ornamentalist.get_config()`
    assert ornamentalist.get_config() == config
```
