"""DINOv2-with-registers image embedding encoder factory.

Wraps ``facebook/dinov2-with-registers-base`` from HuggingFace; uses the
CLS-token embedding from the last hidden state.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from stacking.embed import EncoderFn
from stacking.utils import get_device

MODEL_ID = "facebook/dinov2-with-registers-base"
BATCH_SIZE = 64


def build() -> EncoderFn:
    """Build a DINOv2-with-registers image-embedding closure.

    Returns:
        A callable ``(images) -> np.ndarray[N, D]`` returning the
        CLS-token embedding of each image as a float32 numpy array.
    """
    device = get_device("auto")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    @torch.no_grad()
    def encode(images: Iterable[Image.Image]) -> np.ndarray:
        batch: list[Image.Image] = []
        chunks: list[np.ndarray] = []
        for img in images:
            batch.append(img.convert("RGB"))
            if len(batch) >= BATCH_SIZE:
                chunks.append(_run(batch, processor, model, device))
                batch = []
        if batch:
            chunks.append(_run(batch, processor, model, device))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    return encode


def _run(
    batch: list[Image.Image],
    processor: object,
    model: object,
    device: torch.device,
) -> np.ndarray:
    """Run one batch and return its CLS-token embedding."""
    inputs = processor(images=batch, return_tensors="pt").to(device)  # ty: ignore[unresolved-attribute]
    out = model(**inputs).last_hidden_state[:, 0, :]  # ty: ignore[unresolved-attribute]
    return out.detach().to(torch.float32).cpu().numpy()
