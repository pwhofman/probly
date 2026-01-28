"""TensorFlow data generator implementation.

Runs a Keras model over a tf.data.Dataset, collects simple statistics, and
provides helpers to persist results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type checking only
    import numpy as np

    from .base_generator import BaseDataGenerator
import tensorflow as tf

from .base_generator import BaseDataGenerator

logger = logging.getLogger(__name__)


class TensorFlowDataGenerator(BaseDataGenerator[tf.keras.Model, tf.data.Dataset, str | None]):
    """Data generator for TensorFlow/Keras models."""

    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """Initialize the generator with model, dataset, and runtime options."""
        super().__init__(model, dataset, batch_size, device)

        self.dataset = dataset.batch(batch_size)
        self.device = device or "cpu"
        self.results: dict[str, Any] = {}

        logger.info("TensorFlow DataGenerator initialized")

    def generate(self) -> dict[str, Any]:
        """Run the model on the dataset and compute basic metrics."""
        logger.info("Generating model statistics with TensorFlow...")

        preds_all = []
        labels_all = []

        for i, (x, y) in enumerate(self.dataset):
            preds = self.model(x, training=False)
            preds_all.append(preds)
            labels_all.append(y)

            if (i + 1) % 5 == 0:
                logger.info("Processed %d samples", (i + 1) * self.batch_size)

        preds_all = tf.concat(preds_all, axis=0)
        labels_all = tf.concat(labels_all, axis=0)

        probs = tf.nn.softmax(preds_all, axis=1)
        pred_classes = tf.argmax(probs, axis=1)
        confidences = tf.reduce_max(probs, axis=1)

        accuracy = tf.reduce_mean(
            tf.cast(pred_classes == labels_all, tf.float32),
        ).numpy()

        self.results = {
            "info": {
                "framework": "tensorflow",
                "dataset_size": int(preds_all.shape[0]),
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": float(accuracy),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes.numpy()),
                "ground_truth": self._count(labels_all.numpy()),
            },
            "confidence": {
                "mean": float(tf.reduce_mean(confidences).numpy()),
                "std": float(tf.math.reduce_std(confidences).numpy()),
            },
        }

        return self.results

    def _count(self, values: np.ndarray) -> dict[int, int]:
        counts: dict[int, int] = {}
        for val in values.tolist():
            key = int(val)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def save(self, path: str) -> None:
        """Persist generated results to a JSON file at path."""
        if not self.results:
            logger.info("No results to save.")
            return

        try:
            with Path(path).open("w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            logger.info("Results saved to %s", path)
        except OSError:
            logger.exception("Saving failed")

    def load(self, path: str) -> dict[str, Any]:
        """Load results from a JSON file at path."""
        try:
            with Path(path).open(encoding="utf-8") as f:
                self.results = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.exception("Loading failed")
            self.results = {}
        else:
            logger.info("Results loaded from %s", path)
        return self.results
