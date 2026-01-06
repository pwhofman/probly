"""Data generation utilities and classes."""

# Torch backend exports
# General (pure-Python) backend exports
from .first_order_datagenerator import (
    FirstOrderDataGenerator as GeneralFirstOrderDataGenerator,
    FirstOrderDataset as GeneralFirstOrderDataset,
    output_dataloader as general_output_dataloader,
)

# JAX backend exports
from .jax_first_order_generator import (
    FirstOrderDataGenerator as JaxFirstOrderDataGenerator,
    FirstOrderDataset as JaxFirstOrderDataset,
    output_dataloader as jax_output_dataloader,
)
from .torch_first_order_generator import (
    FirstOrderDataGenerator as TorchFirstOrderDataGenerator,
    FirstOrderDataset as TorchFirstOrderDataset,
    output_dataloader as torch_output_dataloader,
)

__all__ = [
    # Explicit backend-prefixed names only (no ambiguous defaults)
    "GeneralFirstOrderDataGenerator",
    "GeneralFirstOrderDataset",
    "JaxFirstOrderDataGenerator",
    "JaxFirstOrderDataset",
    "TorchFirstOrderDataGenerator",
    "TorchFirstOrderDataset",
    "general_output_dataloader",
    "jax_output_dataloader",
    "torch_output_dataloader",
]
