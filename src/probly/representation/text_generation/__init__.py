"""Text generation representations."""

from __future__ import annotations

from .torch import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTextGenerationSampleSample,
    TorchTokenGeneration,
)

__all__ = [
    "TorchTextGeneration",
    "TorchTextGenerationSample",
    "TorchTextGenerationSampleSample",
    "TorchTokenGeneration",
]
