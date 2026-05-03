"""SigLIP2 image embedding encoder factory.

Wraps ``google/siglip2-base-patch16-256`` from HuggingFace; uses the model's
pooled image embedding.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from stacking.embed import EncoderFn
from stacking.utils import get_device

MODEL_ID = "google/siglip2-base-patch16-256"
BATCH_SIZE = 64


def build() -> EncoderFn:
    """Build a SigLIP2 image-embedding closure.

    Returns:
        A callable ``(images) -> np.ndarray[N, D]`` that batches PIL
        images, runs them through the SigLIP2 vision tower, and returns
        the pooled image embedding as a float32 numpy array.
    """
    device = get_device("auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
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
    """Run one batch and return its pooled embedding as a numpy array."""
    inputs = processor(images=batch, return_tensors="pt").to(device)  # ty: ignore[unresolved-attribute]
    out = model.get_image_features(**inputs)  # ty: ignore[unresolved-attribute]
    return out.detach().to(torch.float32).cpu().numpy()
