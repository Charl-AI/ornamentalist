"""Ornamentalist: decorator-based hyperparameter configuration.

Quick reference (for humans and coding agents):

    import ornamentalist
    from ornamentalist import Configurable

    # 1. Mark configurable parameters with Configurable (or Configurable[default])
    @ornamentalist.configure()
    def train(lr: float = Configurable, epochs: int = Configurable[10]):
        ...

    # 2. For standalone values, use param() instead of a wrapper function
    seed = ornamentalist.param("experiment.seed", int, default=42)

    # 3. Create config and call setup() before any configurable functions
    config = {"train": {"lr": 0.01, "epochs": 20}, "experiment": {"seed": 123}}
    ornamentalist.setup(config)
    train()              # lr=0.01, epochs=20
    seed()               # 123

    # Or generate config from CLI automatically:
    configs = ornamentalist.cli()  # returns a list (see note on sweeps below)
    for config in configs:
        ornamentalist.setup(config, force=True)
        train()

Sharp corners:
  - cli() returns a LIST of configs. Passing multiple values for an argument
    (e.g. --train.lr 0.01 0.001) creates a Cartesian product sweep over all
    combinations. Even with single values, you get a one-element list.
  - Booleans on the CLI are passed as --flag True / --flag False (not --flag / --no-flag).
  - Configurable[default] only provides a default for cli(). When using setup()
    directly, you must always provide every Configurable parameter in the config dict.
  - setup() can only be called once unless you pass force=True.
  - param() returns a callable. Use seed(), not seed, to get the value.
  - Supported CLI types: int, float, bool, str, Literal, and unions with None
    (e.g. int | None, where "null" or "none" on the CLI maps to None).
  - Parameters without type annotations are treated as str by the CLI.
"""

# Written by C Jones, 2025; MIT License.

import argparse
import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import types
import typing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ornamentalist")


__all__ = [
    "disable",
    "setup",
    "cli",
    "get_config",
    "configure",
    "param",
    "Configurable",
    "ConfigDict",
]

# --- types ---

ConfigDict: typing.TypeAlias = dict[str, dict[str, typing.Any]]
"""A nested dict mapping function names to dicts containing
their parameters and values. Input format for ornamentalist.setup().
Example: config = {"my_func": {"param_1": value_1, "param_2": value_2}}
"""


class _NotGiven: ...


@dataclasses.dataclass(frozen=True)
class _Configurable:
    default: typing.Any = _NotGiven

    def __getitem__(self, default):
        return _Configurable(default)


Configurable: typing.Any = _Configurable()
"""Mark arguments as Configurable to tell the configure decorator
about which parameters need to be replaced. Use it as a default
argument for any parameter you wish to be configured by ornamentalist.

To provide a default value for use with ornamentalist.cli(),
use subscript notation, e.g. `param: int = Configurable[123]`.
Default values will have no effect if not using ornamentalist.cli()."""


@dataclasses.dataclass
class _ConfigurableFn:
    name: str  # either original_func.__name__ or the custom name given by the decorator
    original_func: typing.Callable
    params_to_inject: list[str]
    signature: inspect.Signature
    cached_partial: typing.Callable | None = None
    cli_defaults: dict[str, typing.Any] | None = None
    verbose: bool = False

    def __call__(self, *args, **kwargs):
        if _OPERATING_MODE is OperatingMode.DISABLED:
            return self.original_func(*args, **kwargs)
        elif _OPERATING_MODE is OperatingMode.DEFAULT:
            log.warning(
                f"Function {self.original_func.__name__} is decorated with ornamentalist.configure(), "
                "but ornamentalist.setup() has not yet been called to setup the configuration. "
                "Disabling ornamentalist and defaulting to pass-through behaviour. "
                "If this is desired, you can disable this warning by calling ornamentalist.disable()."
            )
            return self.original_func(*args, **kwargs)

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


@dataclasses.dataclass
class _ConfigurableParam:
    group: str
    name: str
    type: type
    default: typing.Any = _NotGiven

    def __call__(self):
        if _OPERATING_MODE is OperatingMode.DISABLED:
            if self.default is not _NotGiven:
                return self.default
            raise ValueError(
                f"ornamentalist is disabled and no default was provided for param '{self.group}.{self.name}'"
            )
        if _OPERATING_MODE is OperatingMode.DEFAULT:
            log.warning(
                f"Param '{self.group}.{self.name}' accessed before ornamentalist.setup() was called. "
                "Defaulting to pass-through behaviour."
            )
            if self.default is not _NotGiven:
                return self.default
            raise ValueError(
                f"ornamentalist.setup() has not been called and no default for param '{self.group}.{self.name}'"
            )
        return get_config()[self.group][self.name]


@dataclasses.dataclass(frozen=True)
class _Cfg:
    config: dict


# --- global state ---


class OperatingMode(enum.Enum):
    DEFAULT = 0
    DISABLED = 1
    ENABLED = 2


_OPERATING_MODE = OperatingMode.DEFAULT
_GLOBAL_CONFIG: _Cfg | None = None
_CONFIG_IS_SET = False
_CONFIGURABLE_FUNCTIONS: list[_ConfigurableFn] = []
_CONFIGURABLE_PARAMS: list[_ConfigurableParam] = []


# --- core ---


def disable():
    """Disable all ornamentalist decorators."""
    global _OPERATING_MODE
    _OPERATING_MODE = OperatingMode.DISABLED


def setup(config: ConfigDict, force: bool = False) -> None:
    """Setup configuration for use in decorated functions.
    Must be called before invoking any decorated functions.

    Raises a ValueError if you try to call setup a second time.
    If this is what you want (i.e. you want to reconfigure
    your functions), you may call setup with Force=True.

    Example:

    ```python
    config = {"my_func": {"param_1": value_1, "param_2": value_2}}
    setup(config)
    ```
    """
    global _GLOBAL_CONFIG, _CONFIG_IS_SET, _OPERATING_MODE
    _OPERATING_MODE = OperatingMode.ENABLED
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
                if p.default.default is not _NotGiven:
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


def param(key: str, type: type, default: typing.Any = _NotGiven) -> _ConfigurableParam:
    """Register a standalone configurable value without needing a wrapper function.

    Usage:
    ```python
        seed = ornamentalist.param("experiment.seed", int, default=42)

        config = {"experiment": {"seed": 123}}
        ornamentalist.setup(config)
        torch.manual_seed(seed())  # returns 123
    ```
    """
    ALLOWED_TYPES = [int, float, bool, str]
    if type not in ALLOWED_TYPES:
        raise ValueError(f"param() only supports types {ALLOWED_TYPES}, got {type}")
    group, name = key.split(".", 1)
    p = _ConfigurableParam(group=group, name=name, type=type, default=default)
    _CONFIGURABLE_PARAMS.append(p)
    return p


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


# enables parsing types such as int | None:
# '2' -> int('2') == 2
# 'null' -> None
def _cast_or_none(v, t):
    if str(v).lower() in ("none", "null"):
        return None
    try:
        return t(v)
    except Exception:
        raise argparse.ArgumentTypeError(f"invalid {t} value: '{v}'")


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
        - int, float, bool, str
        - a Union of any of the above with None (e.g. int | None)
        - a Literal containing concrete values of either str, int, or float.
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

            origin, args = typing.get_origin(anno), typing.get_args(anno)
            if origin is types.UnionType:
                arg_type = [t for t in args if t is not None][0]
                if (
                    len(args) != 2
                    or types.NoneType not in args
                    or arg_type not in [str, int, float]
                ):
                    raise ValueError(
                        "Union types are only supported between int/str/float "
                        f"and None, got {args}."
                    )

                kwargs["type"] = functools.partial(_cast_or_none, t=arg_type)
                kwargs["help"] = f"Type: {arg_type.__qualname__} | None"

            elif origin is typing.Literal:
                if not args:
                    log.warning(
                        f"Parameter '{param_name}' in {f.name} has an empty Literal "
                        "annotation. Skipping."
                    )
                    continue

                arg_type = type(args[0])
                if not all(isinstance(arg, arg_type) for arg in args):
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
                kwargs["choices"] = args
                kwargs["help"] = f"Type: {arg_type.__qualname__}, choices: {args}"

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
                        f"{ALLOWED_TYPES + [typing.Literal]}.\n"
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

    param_groups: dict[str, list[_ConfigurableParam]] = {}
    for p in _CONFIGURABLE_PARAMS:
        param_groups.setdefault(p.group, []).append(p)

    for group_name, params in param_groups.items():
        group = parser.add_argument_group(group_name)
        for p in params:
            kwargs = {"metavar": "\b"}
            kwargs["help"] = f"Type: {p.type.__qualname__}"
            if p.type is bool:
                kwargs["type"] = _str2bool
            else:
                kwargs["type"] = p.type
            if p.default is not _NotGiven:
                kwargs["default"] = p.default
                kwargs["help"] += f" (optional), default={p.default}"
            else:
                kwargs["required"] = True
                kwargs["help"] += " (required)"
            kwargs["nargs"] = "+"
            group.add_argument(f"--{p.group}.{p.name}", **kwargs)

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
