# Ornamentalist

Ornamentalist is a tiny library for configuring functions with fixed hyperparameters in Python. The goal is to allow research code to be more flexible and hackable, without losing readability. It is best summarised by the following sentence:

> What if, instead of threading hyperparameters throughout our programs, we simply replace all configurable functions with `partial` versions of themselves where their hyperparameters are fixed to values given at the start of the program?

The core thing ornamentalist does is it allows you to specify the parameters of a function as `Configurable`. You can then use the `ornamentalist.configure()` decorator to replace the function with a `partial` version of itself. The new partial function has all configurable parameters fixed to values supplied by you at the start of the program. This pattern allows you to avoid the work of plumbing hyperparameters around your code, without resorting to global variables or config God-objects.

You can use ornamentalist alongside your favourite configuration libraries like argparse or hydra. It also comes with the optional `ornamentalist.cli()` feature, which automatically generates a CLI for your program.

I encourage you to read the short [blog post](https://charl-ai.github.io/blog/args) to better understand the motivation behind this libary and why I think ornamentalist is a good solution. For worked examples of how to use ornamentalist with other tools such as hydra, argparse, or submitit, check out the `examples/` directory.

Ornamentalist is only [one file](ornamentalist/__init__.py), so the recommended way to install it is to simply copy-paste it into your projects. You can also install ornamentalist with `pip install ornamentalist`. However, it may lag behind the version available here.

Finally, note that I consider ornamentalist a (mostly) complete piece of software. While I will likely continue to address sharp corners and compatibility, I am unlikely to be overhauling or adding any major features.

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


@ornamentalist.configure(verbose=True)
def add_n(x: int, n: int = Configurable):
    print(x + n)


@ornamentalist.configure(name="greeting_config")
def greet(name: str = Configurable):
    print(f"Hello, {name}")


class MyClass:
    @ornamentalist.configure(name="myclass.init")
    def __init__(self, a: float = Configurable):
        print(a)


if __name__ == "__main__":
    config = {
        "add_n": {"n": 5},
        "greeting_config": {"name": "Alice"},
        "myclass.init": {"a": 4.5},
    }
    ornamentalist.setup(config)

    add_n(10)   # 15
    greet()     # Hello, Alice
    MyClass()   # 4.5
```

By default, ornamentalist uses `func.__name__` as the config key. Use `@configure(name=...)` to override this — useful for class `__init__` methods (which would otherwise all be `"__init__"`), or to avoid name collisions. Setting `verbose=True` logs which parameters are being injected, which is helpful for debugging.

## Default Values

Use `Configurable[value]` to provide a default. Parameters with defaults can be omitted from the config dict; parameters without defaults are required.

```python
@ornamentalist.configure()
def train(lr: float = Configurable, epochs: int = Configurable[10]):
    ...

ornamentalist.setup({"train": {"lr": 0.01}})  # epochs defaults to 10
train()  # lr=0.01, epochs=10
```

Unknown keys in the config are rejected, catching typos early.

## Standalone Parameters

For values that don't belong to a function (e.g. a random seed), use `ornamentalist.param()`:

```python
seed = ornamentalist.param("seed", int, default=42)

ornamentalist.setup({"seed": 123})
torch.manual_seed(seed())  # 123
```

Params are scalar values in the config dict and show up as `--seed` in the CLI. Supported types are `int`, `float`, `bool`, and `str`.

## Automatic CLI

`ornamentalist.cli()` generates an argparse CLI from all registered functions and params. Pass multiple values for any argument to run a grid search over the cartesian product.

```python
@ornamentalist.configure()
def train(lr: float = Configurable, epochs: int = Configurable[10]):
    ...

configs = ornamentalist.cli()
for config in configs:
    ornamentalist.setup(config, force=True)
    train()
```

```
$ python train.py --train.lr 0.01 0.001 --train.epochs 10 20
# runs 4 configs: (0.01, 10), (0.01, 20), (0.001, 10), (0.001, 20)
```

`cli()` returns a list of `ConfigDict` objects — one per combination. See `examples/cli.py` for more detail, including `Literal` types, `int | None` unions, and the auto-generated `--help` output.

## Disabling Ornamentalist

Call `ornamentalist.disable()` to make all decorators pass through without requiring `setup()`. This is useful for tests, notebooks, or importing modules that use ornamentalist without configuring them.

## Examples

- `examples/basics.py` — basic usage as seen above.
- `examples/submitit_basic.py` — launching ornamentalist jobs with submitit.
- `examples/cli.py` — the CLI feature with sweep examples and help output.
- `examples/diffusion_transformer` — a full research codebase using ornamentalist.
