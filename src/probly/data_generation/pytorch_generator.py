"""PyTorch data generator implementation.

Runs a PyTorch model over a dataset, collects simple statistics, and
provides helpers to persist results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .base_generator import BaseDataGenerator

logger = logging.getLogger(__name__)


class PyTorchDataGenerator(BaseDataGenerator[torch.nn.Module, Dataset, str | None]):
    """Data generator for PyTorch models."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
        device: str | None = None,
        num_workers: int = 0,
    ) -> None:
        """Initialize the generator with model, dataset, and runtime options."""
        super().__init__(model, dataset, batch_size, device)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Using device: %s", self.device)

        self.model.to(self.device)
        self.model.eval()

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.results: dict[str, Any] = {}

    def generate(self) -> dict[str, Any]:
        """Run the model on the dataset and compute basic metrics."""
        logger.info("Generating model statistics...")

        outputs_all = []
        labels_all = []

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.dataloader):
                x_device = x_batch.to(self.device)
                out = self.model(x_device)

                outputs_all.append(out.cpu())
                labels_all.append(y_batch.cpu())

                if (i + 1) % 5 == 0:
                    logger.info("Processed %d samples", (i + 1) * self.batch_size)

        outputs = torch.cat(outputs_all, dim=0)
        labels = torch.cat(labels_all, dim=0)

        probs = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(probs, dim=1)
        confidences = torch.max(probs, dim=1).values

        accuracy = (pred_classes == labels).float().mean().item()

        self.results = {
            "info": {
                "framework": "pytorch",
                "dataset_size": len(self.dataset),
                "batch_size": self.batch_size,
            },
            "metrics": {
                "accuracy": accuracy,
                "correct": int((pred_classes == labels).sum().item()),
            },
            "class_distribution": {
                "predicted": self._count(pred_classes),
                "ground_truth": self._count(labels),
            },
            "confidence": {
                "mean": float(confidences.mean()),
                "std": float(confidences.std()),
            },
        }

        return self.results

    def _count(self, tensor: torch.Tensor) -> dict[int, int]:
        counts = {}
        for val in tensor.tolist():
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
