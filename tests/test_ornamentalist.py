import sys
from typing import Literal

import pytest

import ornamentalist
from ornamentalist import Configurable, cli, configure, get_config, param, setup


@pytest.fixture(autouse=True)
def reset_ornamentalist_state():
    registry_before = dict(ornamentalist._REGISTRY)
    yield
    ornamentalist._GLOBAL_CONFIG = None
    ornamentalist._CONFIG_IS_SET = False
    for entry in ornamentalist._REGISTRY.values():
        if isinstance(entry, ornamentalist._ConfigurableFn):
            entry.reset()
    ornamentalist._REGISTRY.clear()
    ornamentalist._REGISTRY.update(registry_before)


@configure()
def basic_func(a: int = Configurable, b: int = Configurable[10]):
    return a + b


@configure(name="custom_name")
def named_func(x: str = Configurable["hello"]):
    return f"{x}, world"


class DecoratedClass:
    @configure(name="my_class")
    def __init__(self, val: float = Configurable):
        self.val = val


@configure()
def literal_func(
    dataset: Literal["cifar", "imagenet"] = Configurable,
    version: Literal[1, 2] = Configurable[1],
):
    return f"{dataset}-v{version}"


@configure()
def no_configurable_params(a: int, b: int):
    """A decorated function with no configurable params to ensure it's a no-op."""
    return a * b


# --- Core Logic Tests ---


def test_setup_and_get_config():
    """Tests that setup correctly populates the config and get_config retrieves it."""
    config = {"basic_func": {"a": 5, "b": 10}}
    setup(config)
    assert get_config() == config


def test_decorated_function_injection():
    """Tests that parameters are correctly injected into a decorated function."""
    config = {"basic_func": {"a": 5, "b": 20}}
    setup(config)
    assert basic_func() == 25


def test_missing_required_param_raises_error():
    """Tests that a ValueError is raised if a required parameter (no default) is missing."""
    config = {"basic_func": {"b": 10}}  # 'a' is required (no default)
    setup(config)
    with pytest.raises(ValueError, match="missing required parameters"):
        basic_func()


def test_default_fills_missing_param():
    """Tests that Configurable[default] fills in missing params in setup()."""
    setup({"basic_func": {"a": 5}})  # 'b' has Configurable[10]
    assert basic_func() == 15


def test_unknown_config_key_raises_error():
    """Tests that an unknown key in the config raises a ValueError (catches typos)."""
    config = {"basic_func": {"a": 5, "b": 10, "typo": 1}}
    setup(config)
    with pytest.raises(ValueError, match="unexpected in config"):
        basic_func()


def test_setup_raises_error_on_resetup():
    """Tests that calling setup twice without force=True raises a ValueError."""
    setup({"basic_func": {"a": 1, "b": 1}})
    with pytest.raises(ValueError, match="Configuration has already been set"):
        setup({"basic_func": {"a": 2, "b": 2}})


def test_setup_force_override():
    """Tests that force=True allows re-configuration and resets functions."""
    setup({"basic_func": {"a": 1, "b": 10}})
    assert basic_func() == 11

    # Re-configure with force=True, providing a new complete config
    setup({"basic_func": {"a": 20, "b": 30}}, force=True)
    assert basic_func() == 50


def test_raises_error_if_not_setup():
    """Tests that calling a configurable function before setup raises a KeyError."""
    with pytest.raises(ValueError, match="Attempted to get config before `setup`"):
        basic_func()


def test_custom_name_decorator():
    """Tests that the @configure(name=...) argument works correctly."""
    config = {"custom_name": {"x": "greetings"}}
    setup(config)
    assert named_func() == "greetings, world"


def test_class_init_decoration():
    """Tests that a class __init__ can be decorated and configured."""
    config = {"my_class": {"val": 3.14}}
    setup(config)
    instance = DecoratedClass()
    assert instance.val == 3.14


def test_no_configurable_params_is_noop():
    """Tests that decorating a function with no Configurable params has no effect."""
    # Note: No setup() call is needed as there's nothing to configure.
    assert no_configurable_params(5, 10) == 50


# --- CLI Tests ---


def test_cli_basic_parsing(monkeypatch):
    """Tests basic CLI argument parsing for a single configuration."""
    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--basic_func.b", "200",
            "--custom_name.x", "goodbye",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar",
            "--literal_func.version", "2",
        ],)
    # fmt: on
    configs = cli()
    assert len(configs) == 1

    assert configs[0] == {
        "basic_func": {"a": 100, "b": 200},
        "custom_name": {"x": "goodbye"},
        "my_class": {"val": 1.0},
        "literal_func": {"dataset": "cifar", "version": 2},
    }


def test_cli_default_value_from_signature(monkeypatch):
    """Tests that Configurable[default] provides the default value to the CLI."""
    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar",
        ],)
    # fmt: on
    configs = cli()
    assert len(configs) == 1

    # fall back to defaults specified in function signature
    # for anything not given on the CLI
    assert configs[0]["basic_func"]["b"] == 10
    assert configs[0]["custom_name"]["x"] == "hello"
    assert configs[0]["literal_func"]["version"] == 1


def test_cli_sweep_parsing(monkeypatch):
    """Tests that providing multiple values generates a Cartesian product of configs."""
    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100", "200",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar", "imagenet",
        ],)
    configs = cli()
    assert len(configs) == 4

    # should have 100,200 x cifar,imagenet (other params are constant)
    expected_configs = [
        {
            "basic_func": {"a": 100, "b": 10},
            "custom_name": {"x": "hello"}, "my_class": {"val": 1.0},
            "literal_func": {"dataset": "cifar", "version": 1},
        },
        {
            "basic_func": {"a": 100, "b": 10},
            "custom_name": {"x": "hello"}, "my_class": {"val": 1.0},
            "literal_func": {"dataset": "imagenet", "version": 1},
        },
        {
            "basic_func": {"a": 200, "b": 10},
            "custom_name": {"x": "hello"}, "my_class": {"val": 1.0},
            "literal_func": {"dataset": "cifar", "version": 1},
        },
        {
            "basic_func": {"a": 200, "b": 10},
            "custom_name": {"x": "hello"}, "my_class": {"val": 1.0},
            "literal_func": {"dataset": "imagenet", "version": 1},
        },
    ]
    # fmt: on
    assert configs == expected_configs


def test_cli_required_argument_missing(monkeypatch):
    """Tests that the CLI exits if a required argument (no default) is missing."""
    monkeypatch.setattr(sys, "argv", ["script.py", "--literal_func.version", "1"])
    with pytest.raises(SystemExit):
        cli()


def test_cli_literal_invalid_choice(monkeypatch):
    """Tests that the CLI exits if an invalid choice for a Literal is given."""
    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "celeba", # invalid
        ],)
    # fmt: on
    with pytest.raises(SystemExit):
        cli()


def test_cli_bool_handling(monkeypatch):
    """Tests the custom boolean handling for CLI arguments."""

    @configure()
    def bool_func(flag: bool = Configurable[False]):
        return flag

    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "imagenet",
            "--literal_func.version", "2",
            "--bool_func.flag", "true",
        ],)
    # fmt: on
    configs = cli()
    assert configs[0]["bool_func"]["flag"] is True

    # Test 'false'
    # fmt: off
    monkeypatch.setattr( # mock giving CLI args to the script
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "imagenet",
            "--literal_func.version", "2",
            "--bool_func.flag", "false",
        ],)
    # fmt: on
    configs = cli()
    assert configs[0]["bool_func"]["flag"] is False


def test_cli_unsupported_type_raises_error():
    """Tests that using an unsupported type hint with cli() raises a ValueError."""

    @configure()
    def bad_type_func(param: list = Configurable):
        pass

    with pytest.raises(
        ValueError, match="Automatic parser generation only works with types"
    ):
        cli()


# --- Param Tests ---


def test_param_basic():
    """Tests that a standalone param resolves its value from the config."""
    seed = param("seed", int)
    setup({"seed": 42})
    assert seed() == 42


def test_param_with_default():
    """Tests that a param with a default can be used, and that the config value takes precedence."""
    seed = param("seed", int, default=0)
    setup({"seed": 99})
    assert seed() == 99


def test_param_missing_config_raises_error():
    """Tests that accessing a param without a default whose key is missing raises a KeyError."""
    seed = param("seed", int)
    setup({"other": {"key": 1}})
    with pytest.raises(KeyError):
        seed()


def test_param_default_fills_missing():
    """Tests that a param with a default resolves when its key is absent from config."""
    seed = param("seed", int, default=42)
    setup({"other": {"key": 1}})
    assert seed() == 42


def test_param_disabled_mode_with_default(monkeypatch):
    """Tests that a param returns its default when ornamentalist is disabled."""
    seed = param("seed", int, default=42)
    monkeypatch.setattr(ornamentalist, "_OPERATING_MODE", ornamentalist.OperatingMode.DISABLED)
    assert seed() == 42


def test_param_disabled_mode_no_default(monkeypatch):
    """Tests that a param without default raises when ornamentalist is disabled."""
    seed = param("seed", int)
    monkeypatch.setattr(ornamentalist, "_OPERATING_MODE", ornamentalist.OperatingMode.DISABLED)
    with pytest.raises(ValueError, match="disabled"):
        seed()


def test_param_default_mode_with_default(monkeypatch):
    """Tests that a param returns its default (with warning) when setup has not been called."""
    monkeypatch.setattr(ornamentalist, "_OPERATING_MODE", ornamentalist.OperatingMode.DEFAULT)
    seed = param("seed", int, default=7)
    assert seed() == 7


def test_param_default_mode_no_default(monkeypatch):
    """Tests that a param without default raises when setup has not been called."""
    monkeypatch.setattr(ornamentalist, "_OPERATING_MODE", ornamentalist.OperatingMode.DEFAULT)
    seed = param("seed", int)
    with pytest.raises(ValueError, match="not been called"):
        seed()


def test_param_unsupported_type():
    """Tests that param() rejects unsupported types."""
    with pytest.raises(ValueError, match="param\\(\\) only supports types"):
        param("data", list)


def test_param_cli(monkeypatch):
    """Tests that standalone params show up in the CLI."""
    seed = param("seed", int)
    # fmt: off
    monkeypatch.setattr(
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar",
            "--seed", "42",
        ],)
    # fmt: on
    configs = cli()
    assert len(configs) == 1
    assert configs[0]["seed"] == 42


def test_param_cli_with_default(monkeypatch):
    """Tests that a param with a default is optional in the CLI."""
    seed = param("seed", int, default=7)
    # fmt: off
    monkeypatch.setattr(
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar",
        ],)
    # fmt: on
    configs = cli()
    assert configs[0]["seed"] == 7


def test_param_cli_sweep(monkeypatch):
    """Tests that params participate in grid search sweeps."""
    seed = param("seed", int)
    # fmt: off
    monkeypatch.setattr(
        sys, "argv", [
            "script.py",
            "--basic_func.a", "100",
            "--my_class.val", "1.0",
            "--literal_func.dataset", "cifar",
            "--seed", "1", "2", "3",
        ],)
    # fmt: on
    configs = cli()
    assert len(configs) == 3
    seeds = [c["seed"] for c in configs]
    assert seeds == [1, 2, 3]


# --- Collision Detection Tests ---


def test_function_name_collision():
    """Tests that registering two functions with the same name raises ValueError."""
    @configure(name="duplicate")
    def func_a(x: int = Configurable):
        return x

    with pytest.raises(ValueError, match="already registered"):
        @configure(name="duplicate")
        def func_b(y: int = Configurable):
            return y


def test_param_name_collision():
    """Tests that registering two params with the same name raises ValueError."""
    param("seed", int)
    with pytest.raises(ValueError, match="already registered"):
        param("seed", float)


def test_function_param_cross_collision():
    """Tests that a param can't use the same name as a configured function."""
    @configure(name="train")
    def train_fn(lr: float = Configurable):
        return lr

    with pytest.raises(ValueError, match="conflicts with a configured function"):
        param("train", int)


def test_param_function_cross_collision():
    """Tests that a function can't use the same name as a registered param."""
    param("seed", int)
    with pytest.raises(ValueError, match="conflicts with param"):
        @configure(name="seed")
        def seed_fn(x: int = Configurable):
            return x


# --- Config Immutability Tests ---


def test_get_config_immutable_top_level():
    """Tests that the top-level config dict returned by get_config() is immutable."""
    setup({"basic_func": {"a": 1, "b": 2}})
    config = get_config()
    with pytest.raises(TypeError):
        config["new_key"] = "bad"


def test_get_config_immutable_nested():
    """Tests that nested dicts in get_config() are immutable."""
    setup({"basic_func": {"a": 1, "b": 2}})
    config = get_config()
    with pytest.raises(TypeError):
        config["basic_func"]["a"] = 999
