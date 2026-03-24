"""JAX data generator implementation.

Runs a JAX model over input arrays, collects simple statistics, and
provides helpers to persist results.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

import jax
import jax.numpy as jnp

from .base_generator import BaseDataGenerator

logger = logging.getLogger(__name__)


class JAXDataGenerator(BaseDataGenerator[Callable[[jnp.ndarray], jnp.ndarray], tuple[object, object], str | None]):
    """Data generator for JAX models."""

    def __init__(
        self,
        model: Callable[[jnp.ndarray], jnp.ndarray],  # JAX model function (callable)
        dataset: tuple[object, object],  # (x, y) as numpy or jax arrays
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Initialize the generator with model, dataset, and runtime options."""
        super().__init__(model, dataset, batch_size, device)

        self.device = device or "cpu"
        self.results: dict[str, Any] = {}

        logger.info("JAX DataGenerator initialized")

    def generate(self) -> dict[str, Any]:
        """Run the model on the dataset and compute basic metrics."""
        logger.info("Generating model statistics with JAX...")

        x, y = self.dataset
        x = jnp.array(x)
        y = jnp.array(y)

        # forward pass (assume classification logits)
        logits = self.model(x)
        probs = jax.nn.softmax(logits, axis=1)

        pred_classes = jnp.argmax(probs, axis=1)
        confidences = jnp.max(probs, axis=1)

        accuracy = jnp.mean(pred_classes == y)

        self.results = {
            "info": {
                "framework": "jax",
                "dataset_size": x.shape[0],
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": float(accuracy),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes),
                "ground_truth": self._count(y),
            },
            "confidence": {
                "mean": float(jnp.mean(confidences)),
                "std": float(jnp.std(confidences)),
            },
        }

        return self.results

    def _count(self, values: jnp.ndarray) -> dict[int, int]:
        values = jnp.asarray(values).astype(jnp.int32)
        bc = jnp.bincount(values).tolist()
        return {i: int(c) for i, c in enumerate(bc) if c != 0}
