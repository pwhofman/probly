"""Data generation utilities and classes."""

import sys as _sys

# General (pure-Python) backend exports
from .first_order_datagenerator import (
    FirstOrderDataGenerator as GeneralFirstOrderDataGenerator,
    FirstOrderDataset as GeneralFirstOrderDataset,
    output_dataloader as general_output_dataloader,
)

# Torch backend exports
from .torch_first_order_generator import (
    FirstOrderDataGenerator as TorchFirstOrderDataGenerator,
    FirstOrderDataset as TorchFirstOrderDataset,
    output_dataloader as torch_output_dataloader,
)

# JAX backend exports (conditionally available)
try:
    from .jax_first_order_generator import (
        FirstOrderDataGenerator as JaxFirstOrderDataGenerator,
        FirstOrderDataset as JaxFirstOrderDataset,
        output_dataloader as jax_output_dataloader,
    )
    _JAX_AVAILABLE = True
except ModuleNotFoundError:
    # If running under pytest, skip JAX-dependent tests at module import.
    if "pytest" in _sys.modules:
        import pytest as _pytest

        _pytest.skip(
            "JAX not installed so skipping JAX-dependent tests.",
            allow_module_level=True,
        )
    _JAX_AVAILABLE = False

__all__ = [
    # Explicit backend-prefixed names only (no ambiguous defaults)
    "GeneralFirstOrderDataGenerator",
    "GeneralFirstOrderDataset",
    "TorchFirstOrderDataGenerator",
    "TorchFirstOrderDataset",
    "general_output_dataloader",
    "torch_output_dataloader",
]

if _JAX_AVAILABLE:
    __all__.extend([
        "JaxFirstOrderDataGenerator",
        "JaxFirstOrderDataset",
        "jax_output_dataloader",
    ])
