"""TensorFlow data generator implementation.

Runs a Keras model over a tf.data.Dataset, collects simple statistics, and
provides helpers to persist results.
"""

from __future__ import annotations

import logging
from typing import Any

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
                "predicted": self._count(pred_classes),
                "ground_truth": self._count(labels_all),
            },
            "confidence": {
                "mean": float(tf.reduce_mean(confidences).numpy()),
                "std": float(tf.math.reduce_std(confidences).numpy()),
            },
        }

        return self.results

    def _count(self, values: tf.Tensor) -> dict[int, int]:
        values = tf.cast(values, tf.int32)
        bc = tf.math.bincount(values).numpy().tolist()
        return {i: int(c) for i, c in enumerate(bc) if c != 0}
