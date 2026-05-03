"""Embedding-cache I/O and lazy encoder registry.

Encoder factories are imported lazily so the heavy ``transformers`` stack is
only loaded when the cache script actually needs an encoder. The cifar10h
loader only needs :func:`cache_path`, never the encoders themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image

EncoderFn = Callable[[Iterable[Image.Image]], np.ndarray]
"""Takes an iterable of PIL images, returns a (N, D) float32 numpy array."""


def cache_path(*, encoder: str, dataset: str, split: str, root: Path) -> Path:
    """Return the on-disk cache path for one (dataset, encoder, split).

    Args:
        encoder: Encoder name (e.g. ``"siglip2"``).
        dataset: Dataset key. Use ``"cifar10"`` for the train cache and
            ``"cifar10h"`` for the test cache.
        split: Split name (``"train"`` or ``"test"``).
        root: Root cache directory.

    Returns:
        ``<root>/<dataset>_<encoder>_<split>.npz``.
    """
    return root / f"{dataset}_{encoder}_{split}.npz"


def _build_siglip2() -> EncoderFn:
    """Build the SigLIP2 image embedding closure (filled in Task 8)."""
    from stacking._encoders.siglip2 import build  # noqa: PLC0415

    return build()


def _build_dinov2_with_registers() -> EncoderFn:
    """Build the DINOv2-with-registers image embedding closure (Task 8)."""
    from stacking._encoders.dinov2_with_registers import build  # noqa: PLC0415

    return build()


ENCODERS: dict[str, Callable[[], EncoderFn]] = {
    "siglip2": _build_siglip2,
    "dinov2_with_registers": _build_dinov2_with_registers,
}
"""Registry mapping encoder name to a zero-arg factory returning an EncoderFn."""
