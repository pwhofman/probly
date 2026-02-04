"""Implementations for Vector-, Platt- and Temperature Scaling."""

from __future__ import annotations

from typing import Any

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from .common import _platt_factory, _temperature_factory, _vector_factory


# Torch
@_temperature_factory.register(TORCH_MODULE)
def _(_base: object) -> type[Any]:
    from . import torch_temperature  # noqa: F401, PLC0415

    return _temperature_factory(_base)


@_platt_factory.register(TORCH_MODULE)
def _(_base: object) -> type[Any]:
    from . import torch_platt  # noqa: F401, PLC0415

    return _platt_factory(_base)


@_vector_factory.register(TORCH_MODULE)
def _(_base: object, _num_classes: int) -> type[Any]:
    from . import torch_vector  # noqa: F401, PLC0415

    return _vector_factory(_base, _num_classes)


# Flax
@_temperature_factory.register(FLAX_MODULE)
def _(_base: object) -> type[Any]:
    from . import flax_temperature  # noqa: F401, PLC0415

    return _temperature_factory(_base)


@_platt_factory.register(FLAX_MODULE)
def _(_base: object) -> type[Any]:
    from . import flax_platt  # noqa: F401, PLC0415

    return _platt_factory(_base)


@_vector_factory.register(FLAX_MODULE)
def _(_base: object, _num_classes: int) -> type[Any]:
    from . import flax_vector  # noqa: F401, PLC0415

    return _vector_factory(_base, _num_classes)
