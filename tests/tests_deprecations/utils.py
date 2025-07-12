"""A module containing all deprecated features / behaviors in probly. Adapted from the shapiq package.

Usage:
    To add a new deprecated feature or behavior, create an instance of the `DeprecatedFeature` and
    append it to the `DEPRECATED_FEATURES` list. The `call` attribute should be a callable that
    triggers the deprecation warning when executed. The `deprecated_in` and `removed_in` attributes
    should specify the version in which the feature was deprecated and the version in which it will
    be removed, respectively.

"""

from __future__ import annotations

import importlib
import pathlib
import pkgutil
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

    DeprecatedTestFunc = Callable[[pytest.FixtureRequest], None]


class DeprecatedFeature(NamedTuple):
    """A named tuple to represent a deprecated feature."""

    name: str
    deprecated_in: str
    removed_in: str
    call: Callable[[pytest.FixtureRequest], None]


DEPRECATED_FEATURES: list[DeprecatedFeature] = []


def register_deprecated(
    name: str,
    deprecated_in: str,
    removed_in: str,
) -> Callable[[DeprecatedTestFunc], DeprecatedTestFunc]:
    """Decorator to register a deprecated feature.

    Args:
        name: The name of the deprecated feature.
        deprecated_in: The version in which the feature was deprecated.
        removed_in: The version in which the feature will be removed.

    Returns:
        A decorator that registers the deprecated feature.
    """

    def _decorator(func: Callable[[pytest.FixtureRequest], None]) -> Callable[[pytest.FixtureRequest], None]:
        DEPRECATED_FEATURES.append(DeprecatedFeature(name, deprecated_in, removed_in, func))
        return func

    return _decorator


# auto-import all deprecated modules from this folder in the current package
def _auto_import_deprecated_modules() -> None:
    current_path = pathlib.Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(current_path)]):
        name = module_info.name
        if name not in {"__init__", "features"}:
            importlib.import_module(f"{__package__}.{name}")


_auto_import_deprecated_modules()
