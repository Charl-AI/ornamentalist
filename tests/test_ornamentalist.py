import sys
from typing import Literal

import pytest

import ornamentalist
from ornamentalist import Configurable, cli, configure, get_config, setup


@pytest.fixture(autouse=True)
def reset_ornamentalist_state():
    yield
    ornamentalist._GLOBAL_CONFIG = None
    ornamentalist._CONFIG_IS_SET = False
    for f in ornamentalist._CONFIGURABLE_FUNCTIONS:
        f.reset()


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


def test_incomplete_config_raises_error():
    """Tests that a ValueError is raised if the config is missing a required
    parameter, confirming we don't silently fall back to defaults."""
    config = {"basic_func": {"a": 5}}  # 'b' is missing
    setup(config)
    with pytest.raises(ValueError, match="parameters injected by config do not match"):
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
