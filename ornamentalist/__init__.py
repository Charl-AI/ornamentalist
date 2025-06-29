"""Ornamentalist is a library for decorator-based hyperparameter configuration."""

# Written by C Jones, 2025; MIT License.

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ornamentalist")


__all__ = ["setup_config", "get_config", "configure", "Configurable"]


@dataclass(frozen=True)
class _Cfg:
    config: dict


_GLOBAL_CONFIG: _Cfg | None = None
_CONFIG_IS_SET = False
# tracks whether we have tried to call any configurable
# functions yet (not safe to change config after this)
_ORNAMENTALIST_USED = False


def setup_config(config: dict, force: bool = False) -> None:
    """Setup configuration for use in decorated functions.
    Must be called before any decorated functions.

    `config` is expected to be a nested dictionary mapping
    function names to dictionaries containing their args and
    configuration values.

    Examples:

    ```python
    config = {"my_function": {"param_1": value, "param_2": value2}}
    setup_config(config)
    ```
    """
    global _GLOBAL_CONFIG, _CONFIG_IS_SET
    if _ORNAMENTALIST_USED:
        raise ValueError(
            "Changing configuration after calling a configured function is not supported."
        )
    if _CONFIG_IS_SET and not force:
        raise ValueError(
            "Configuration has already been set. Use force=True to override."
        )

    c = _Cfg(config)
    _GLOBAL_CONFIG = c
    _CONFIG_IS_SET = True


def get_config() -> dict:
    if _GLOBAL_CONFIG is None or not _CONFIG_IS_SET:
        raise ValueError("Attempted to get config before `setup_config` is called.")
    return _GLOBAL_CONFIG.config


class _Configurable:
    def __repr__(self):
        return "<CONFIGURABLE_PARAM>"


"""Mark arguments as Configurable to tell the configure decorator
about which parameters need to be replaced. Use it as a default
argument for any parameter you wish to be configured by ornamentalist."""
Configurable: Any = _Configurable()


def configure(name: str | None = None, verbose: bool = False):
    """Decorate a function with @configure() to inject
    replace some or all of its arguments with values from
    your program configuration."""

    def decorator(func):
        _cached_partial = None

        nonlocal name
        name = name if name is not None else func.__name__

        signature = inspect.signature(func)
        params_to_inject = [
            p.name for p in signature.parameters.values() if p.default is Configurable
        ]

        if not params_to_inject:
            if verbose:
                log.info("No Configurable parameters found, returning function as-is.")
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _cached_partial
            if _cached_partial is None:
                injected_params = get_config()[name]
                if verbose:
                    log.info(
                        msg=f"Injecting parameters {injected_params} into {func.__name__}"
                    )

                    if set(injected_params.keys()) != set(params_to_inject):
                        raise ValueError(
                            "Parameters injected by config do not match "
                            + "the parameters marked as Configurable: "
                            + f"{set(injected_params)=} != {set(params_to_inject)=}"
                        )

                _cached_partial = functools.partial(func, **injected_params)
                global _ORNAMENTALIST_USED
                _ORNAMENTALIST_USED = True
            return _cached_partial(*args, **kwargs)

        return wrapper

    return decorator
