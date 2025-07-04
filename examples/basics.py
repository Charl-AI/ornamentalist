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
