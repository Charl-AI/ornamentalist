"""Ornamentalist is a library for decorator-based hyperparameter configuration."""

# Written by C Jones, 2025; MIT License.

import argparse
import dataclasses
import functools
import inspect
import itertools
import logging
from typing import Any, Callable, Literal, TypeAlias, get_args

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ornamentalist")


__all__ = ["setup", "cli", "get_config", "configure", "Configurable"]

# --- types ---

ConfigDict: TypeAlias = dict[str, dict[str, Any]]
"""A nested dict mapping function names to dicts containing
their parameters and values. Input format for ornamentalist.setup().
Example: config = {"my_func": {"param_1": value_1, "param_2": value_2}}
"""


@dataclasses.dataclass(frozen=True)
class _Configurable:
    default: Any = None

    def __getitem__(self, default):
        return _Configurable(default)


Configurable: Any = _Configurable()
"""Mark arguments as Configurable to tell the configure decorator
about which parameters need to be replaced. Use it as a default
argument for any parameter you wish to be configured by ornamentalist.
To provide a default value for use with ornamentalist.cli(),
use subscript notation, e.g. `param: int = Configurable[123]`."""


@dataclasses.dataclass
class _ConfigurableFn:
    name: str  # either original_func.__name__ or the custom name given by the decorator
    original_func: Callable
    params_to_inject: list[str]
    signature: inspect.Signature
    cached_partial: Callable | None = None
    cli_defaults: dict[str, Any] | None = None
    verbose: bool = False

    def __call__(self, *args, **kwargs):
        if self.cached_partial is None:
            fn_name = (
                f"{self.original_func.__module__}.{self.original_func.__qualname__}"
            )
            config = get_config()
            if self.name not in config:
                raise KeyError(
                    f"Configuration for '{self.name}' not found in global config, got {get_config()=}"
                )
            injected_params = config[self.name]
            if self.verbose:
                log.info(msg=f"Injecting parameters {injected_params} into {fn_name}")

            if set(injected_params.keys()) != set(self.params_to_inject):
                raise ValueError(
                    f"Tried to inject parameters into {fn_name}, but "
                    + "parameters injected by config do not match "
                    + "the parameters marked as Configurable:\n"
                    + f"{set(injected_params)=} != {set(self.params_to_inject)=}"
                )

            self.cached_partial = functools.partial(
                self.original_func, **injected_params
            )
        return self.cached_partial(*args, **kwargs)

    def reset(self):
        self.cached_partial = None


@dataclasses.dataclass(frozen=True)
class _Cfg:
    config: dict


# --- global state ---


_GLOBAL_CONFIG: _Cfg | None = None
_CONFIG_IS_SET = False
_CONFIGURABLE_FUNCTIONS: list[_ConfigurableFn] = []


# --- core ---


def setup(config: ConfigDict, force: bool = False) -> None:
    """Setup configuration for use in decorated functions.
    Must be called before invoking any decorated functions.

    The config dict should take the form of a nested dict,
    mapping function names to dicts containing each function's

    Raises a ValueError if you try to call setup a second time.
    If this is what you want (i.e. you want to reconfigure
    your functions), you may call setup with Force=True.

    Example:

    ```python
    config = {"my_func": {"param_1": value_1, "param_2": value_2}}
    setup(config)
    ```
    """
    global _GLOBAL_CONFIG, _CONFIG_IS_SET
    if _CONFIG_IS_SET and not force:
        raise ValueError(
            "Configuration has already been set. Use force=True to override."
        )

    if force:
        for f in _CONFIGURABLE_FUNCTIONS:
            f.reset()

    if not config:
        log.warning("The configuration is empty. No parameters will be injected.")

    c = _Cfg(config)
    _GLOBAL_CONFIG = c
    _CONFIG_IS_SET = True


def get_config() -> ConfigDict:
    """Returns the ConfigDict that you used in ornamentalist.setup()."""
    if _GLOBAL_CONFIG is None or not _CONFIG_IS_SET:
        raise ValueError("Attempted to get config before `setup` has been called.")
    return _GLOBAL_CONFIG.config


def configure(name: str | None = None, verbose: bool = False):
    """Decorate a function with @configure() to replace all Configurable arguments
    with values from your program configuration.

    Usage:
    ```python
        @ornamentalist.configure()
        def parametric_fn(x: int, param: float = Configurable):
            ... # does something with x and param

        config = {"parametric_fn": {"param": 1.5}}
        setup(config)
        parametric_fn(x=2) # param is now set to 1.5, so we don't pass it here

    ```
    You can think of this decorator as lazily creating a partial function. I.e.
    the above example is approximately equivalent to:

    ```python
        parametric_fn = functools.partial(parametric_fn, param=1.5)
        parametric_fn(x=2)

    ```

    """

    def decorator(func):
        nonlocal name
        name = name if name is not None else func.__name__

        signature = inspect.signature(func)

        params_to_inject = []
        cli_defaults = {}
        for p in signature.parameters.values():
            if isinstance(p.default, _Configurable):
                params_to_inject.append(p.name)
                if p.default.default is not None:
                    cli_defaults[p.name] = p.default.default

        if not params_to_inject:
            if verbose:
                log.info("No Configurable parameters found, returning function as-is.")
            return func

        configurable_fn = _ConfigurableFn(
            original_func=func,
            name=name,
            params_to_inject=params_to_inject,
            signature=signature,
            verbose=verbose,
            cli_defaults=cli_defaults,
        )
        _CONFIGURABLE_FUNCTIONS.append(configurable_fn)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return configurable_fn(*args, **kwargs)

        return wrapper

    return decorator


# --- cli (optional extra feature) ---


# we (controversially) use this to enable argparse to handle --arg=True and
# --arg=False properly. Without it, any non-empty arg evaluates to True.
# The canonical way of doing this in argparse is to have --arg and --no-arg
# flags, with store_true and store_false actions, respectively. However,
# I prefer our way because it more consistent with the other args. It
# also makes it easier to sweep over both configs with '--arg True False'
def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t"):
        return True
    elif v.lower() in ("false", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _namespace2dict(config_ns: argparse.Namespace) -> ConfigDict:
    config_dict = {}
    for flat_key, value in vars(config_ns).items():
        if "." in flat_key:
            fn_name, param_name = flat_key.split(".", 1)
            if fn_name not in config_dict:
                config_dict[fn_name] = {}
            config_dict[fn_name][param_name] = value
    return config_dict


def cli(parser: argparse.ArgumentParser | None = None) -> list[ConfigDict]:
    """Automatically generates a CLI for your program using argparse.
    All functions marked with ornamentalist.configure() will have Configurable
    parameters show up as options in the CLI. If you want to add extra
    CLI parameters, you can pass a pre-existing argparse parser to this function.

    The returned ConfigDict object(s) can be passed to
    ornamentalist.setup() to configure your program.

    Note that automatic CLI generation only works if Configurable parameters
    are annotated with one of the following built-in types:
        - int
        - float
        - bool
        - str
        - Literal[str|int|float]
    If you do not use type annotations in your function signatures, argparse
    will default to treating them as strings. You will then have to manually
    deal with casting them to whatever type you wish to use.

    The CLI comes with support for grid search over hyperparameters.
    Simply provide a list of values for each argument on the command line.
    The function will then return a list of ConfigDict objects
    corresponding to the cartesian product of all arguments given.

    Usage:
    ```python

        @ornamentalist.configure()
        def parametric_fn(x: int, param: float = Configurable):
            ... # does something with x and param

        configs = ornamentalist.cli()

        # running sweep in a loop... you can also delegate to
        # other processes or jobs using submitit etc.
        for config in configs:
            ornamentalist.setup(args)
            parametric_fn(x=2)

        # run with python script.py --parametric_fn.param 1.5 2.0 2.5

    ```

    """
    ALLOWED_TYPES = [int, float, bool, str]
    if parser is None:
        parser = argparse.ArgumentParser()

    for f in _CONFIGURABLE_FUNCTIONS:
        fn_name = f"{f.original_func.__module__}.{f.original_func.__qualname__}"
        group = parser.add_argument_group(
            f.name, description=f"Hyperparameters for {fn_name}"
        )

        for param_name in f.params_to_inject:
            param = f.signature.parameters[param_name]
            assert isinstance(param.default, _Configurable)
            anno = param.annotation
            kwargs = {}
            kwargs["metavar"] = "\b"

            is_literal = hasattr(anno, "__origin__") and anno.__origin__ is Literal
            if is_literal:
                literal_args = get_args(anno)
                if not literal_args:
                    log.warning(
                        f"Parameter '{param_name}' in {f.name} has an empty Literal "
                        "annotation. Skipping."
                    )
                    continue

                arg_type = type(literal_args[0])
                if not all(isinstance(arg, arg_type) for arg in literal_args):
                    raise ValueError(
                        f"All choices in Literal for '{param_name}' in {f.name} "
                        "must be of the same type."
                    )

                if arg_type not in [str, int, float]:
                    raise ValueError(
                        f"Literal type for '{param_name}' in {f.name} must contain "
                        f"str, int, or float, but got {arg_type}."
                    )

                kwargs["type"] = arg_type
                kwargs["choices"] = literal_args
                kwargs["help"] = (
                    f"Type: {arg_type.__qualname__}, choices: {literal_args}"
                )

            else:
                if anno is inspect.Parameter.empty:
                    msg = (
                        f"No type annotation was provided for '{param_name}' "
                        + f"in function {fn_name}.\nArgparse will default to treating "
                        + "it as a string and you will have to manually cast to other types."
                    )
                    log.warning(msg)

                if anno not in ALLOWED_TYPES and anno is not inspect.Parameter.empty:
                    msg = (
                        f"Tried to create a parser for {fn_name}, but "
                        + f"parameter '{param_name}' has type annotation '{anno}'.\n"
                        + "Automatic parser generation only works with types: "
                        f"{ALLOWED_TYPES + [Literal]}.\n"
                        + "(If you provide no annotation, argparse will treat it as 'str')"
                    )
                    raise ValueError(msg)

                type_name = (
                    anno.__qualname__
                    if anno is not inspect.Parameter.empty
                    else "Unknown [string fallback]"
                )
                kwargs["help"] = f"Type: {type_name}"

                if anno is bool:
                    kwargs["type"] = _str2bool
                elif anno is not inspect.Parameter.empty:
                    kwargs["type"] = anno

            if f.cli_defaults is not None and param_name in f.cli_defaults:
                kwargs["default"] = f.cli_defaults[param_name]
                kwargs["help"] += f" (optional), default={f.cli_defaults[param_name]}"
            else:
                kwargs["required"] = True
                kwargs["help"] += " (required)"

            kwargs["nargs"] = "+"
            group.add_argument(f"--{f.name}.{param_name}", **kwargs)

    args_dict = vars(parser.parse_args())
    param_names = sorted(args_dict.keys())
    value_lists = []
    for name in param_names:
        value = args_dict[name]
        if not isinstance(value, list):
            value_lists.append([value])
        else:
            value_lists.append(value)
    product = itertools.product(*value_lists)

    configs = []
    for combo in product:
        config_ns = argparse.Namespace(**dict(zip(param_names, combo)))
        configs.append(_namespace2dict(config_ns))

    return configs
