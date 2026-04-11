"""Implementations for Vector-, Platt- and Temperature Scaling."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from .common import (
    _platt_factory,
    _temperature_factory,
    _vector_factory,
    platt,
    register_platt_factory,
    register_vector_factory,
    temperature,
    vector,
)

__all__ = [
    "platt",
    "register_platt_factory",
    "register_temperature_factory",
    "register_vector_factory",
    "temperature",
    "vector",
]


# Torch
@_temperature_factory.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_temperature  # noqa: F401, PLC0415


@_platt_factory.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_platt  # noqa: F401, PLC0415


@_vector_factory.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch_vector  # noqa: F401, PLC0415


# Flax
@_temperature_factory.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax_temperature  # noqa: F401, PLC0415


@_platt_factory.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax_platt  # noqa: F401, PLC0415


@_vector_factory.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    from . import flax_vector  # noqa: F401, PLC0415
