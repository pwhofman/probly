"""Factory for creating framework-specific data generators.

Selects an appropriate BaseDataGenerator implementation based on the
framework argument ("pytorch", "tensorflow", or "jax").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from .jax_generator import JAXDataGenerator
from .pytorch_generator import PyTorchDataGenerator

if TYPE_CHECKING:  # import for type checking only
    from collections.abc import Callable

    import torch
    from torch.utils.data import Dataset as TorchDataset

    from .base_generator import BaseDataGenerator


@overload
def create_data_generator(
    framework: Literal["pytorch"],
    model: torch.nn.Module,
    dataset: TorchDataset[Any],
    batch_size: int = 32,
    device: str | None = None,
) -> PyTorchDataGenerator: ...


@overload
def create_data_generator(
    framework: Literal["jax"],
    model: Callable[[Any], Any],
    dataset: tuple[Any, Any],
    batch_size: int = 32,
    device: str | None = None,
) -> JAXDataGenerator: ...


def create_data_generator(
    framework: str,
    model: Any,
    dataset: Any,
    batch_size: int = 32,
    device: str | None = None,
) -> BaseDataGenerator[Any, Any, str | None]:
    """Create a data generator based on the selected framework."""
    framework = framework.lower()

    if framework == "pytorch":
        return PyTorchDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    if framework == "jax":
        return JAXDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    msg = f"Unknown framework: {framework}"
    raise ValueError(msg)
