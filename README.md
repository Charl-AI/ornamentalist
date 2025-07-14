# Ornamentalist

Ornamentalist is a tiny library for configuring functions with fixed hyperparameters in Python. The goal is to allow research code to be more flexible and hackable, without losing readability.

The core thing ornamentalist does is it allows you to specify the parameters of a function as `Configurable`. You can then use the `ornamentalist.configure()` decorator to replace the function with a `partial` version of itself. The new partial function has all configurable parameters fixed to values supplied by you at the start of the program. This pattern allows you to avoid the work of plumbing hyperparameters around your code, without resorting to global variables or config God-objects.

You can use ornamentalist alongside your favourite configuration libraries like argparse or hydra. We also provide the optional `ornamentalist.cli()` feature, which automatically generates a CLI for your program.

I encourage you to read the short [blog post](https://charl-ai.github.io/blog/args) to better understand the motivation behind this libary and why I think ornamentalist is a good solution. For worked examples of how to use ornamentalist with other tools such as hydra, argparse, or submitit, check out the `examples/` directory.

You can install ornamentalist with pip:

```
pip install ornamentalist
```

Ornamentalist is only 1-file, so feel free to copy-paste it into your projects if you prefer.

## Usage

Using ornamentalist is straightforward:

1. Mark hyperparameters as configurable by setting their default value to `ornamentalist.Configurable`.
2. Decorate the function with `@ornamentalist.configure()`.
3. Create a config dictionary at the start of your program (either with `ornamentalist.cli()` or your favourite configuration tool).
4. Call `ornamentalist.setup(config)` before running any configurable functions.

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
```

## Examples

Ornamentalist is a simple library! You can learn the whole thing by reading through these examples:

- `examples/basics.py` teaches you the basic usage of ornamentalist (as seen above).
- `examples/submitit_basic.py` shows you a simple pattern for launching ornamentalist jobs with submitit.
- `examples/cli.py` demonstrates how to use the automatic CLI generation feature (with example outputs).
- `examples/diffusion_transformer` is an example of a full research codebase using ornamentalist.