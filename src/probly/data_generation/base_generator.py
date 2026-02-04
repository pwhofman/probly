"""Data generation base interfaces.

This file defines the abstract BaseDataGenerator interface used to run a
model over a dataset, collect statistics, and persist results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any


class BaseDataGenerator[M, D, Dev](ABC):
    """Base class for data generators."""

    def __init__(
        self,
        model: M,
        dataset: D,
        batch_size: int = 32,
        device: Dev | None = None,
    ) -> None:
        """Initialize the data generator.

        - model: The predictive model to evaluate.
        - dataset: The dataset or dataloader to iterate over.
        - batch_size: Number of samples processed per batch.
        - device: Optional execution device information.
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def generate(self) -> dict[str, Any]:
        """Run the model on the dataset and collect stats."""

    def save(self, path: str) -> None:
        """Save generated results to a file."""
        results = self.generate()

        with Path.open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    def load(self, path: str) -> dict[str, Any]:
        """Load results from a file."""
        with Path.open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_info(self) -> dict[str, Any]:
        """Return a summary of the generator configs."""
        return {
            "batch_size": self.batch_size,
            "device": self.device,
        }
