# noqa: INP001
"""Factory for creating framework-specific data generators.

Selects an appropriate BaseDataGenerator implementation based on the
framework argument ("pytorch", "tensorflow", or "jax").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .jax_generator import JAXDataGenerator
from .pytorch_generator import PyTorchDataGenerator
from .tensorflow_generator import TensorFlowDataGenerator

if TYPE_CHECKING:  # import for type checking only
    from .base_generator import BaseDataGenerator


def create_data_generator(
    framework: str,
    model: object,
    dataset: object,
    batch_size: int = 32,
    device: str | None = None,
) -> BaseDataGenerator[object, object, str | None]:
    """Create a data generator based on the selected framework."""
    framework = framework.lower()

    if framework == "pytorch":
        return PyTorchDataGenerator(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
        )

    if framework == "tensorflow":
        return TensorFlowDataGenerator(
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
