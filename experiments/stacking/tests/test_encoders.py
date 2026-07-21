"""Encoder factory smoke tests.

These tests are slow (they download a HuggingFace checkpoint) and skipped
by default. Run them explicitly with:

    STACKING_TEST_ENCODERS=1 uv run pytest tests/test_encoders.py -v
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
import pytest

from stacking.embed import ENCODERS

requires_encoders = pytest.mark.skipif(
    os.environ.get("STACKING_TEST_ENCODERS", "0") != "1",
    reason="set STACKING_TEST_ENCODERS=1 to download and run encoder factories",
)


@requires_encoders
@pytest.mark.parametrize("name", ["siglip2", "dinov2_with_registers"])
def test_encoder_returns_float32_2d_array(name: str) -> None:
    """Each encoder returns a (N, D) float32 array on tiny dummy images."""
    encoder = ENCODERS[name]()
    images = [Image.new("RGB", (32, 32), color=(i, 0, 0)) for i in range(3)]
    out = encoder(images)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 2
    assert out.shape[0] == 3
    assert out.shape[1] > 0
