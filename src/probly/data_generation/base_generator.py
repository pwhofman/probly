from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataGenerator(ABC):
    """Base class for data generators used in our project."""

    def __init__(self, model, dataset, batch_size=32, device=None):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def generate(self) -> dict[str, Any]:
        """Run the model on the dataset and collect statistics."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save generated results to a file."""

    @abstractmethod
    def load(self, path: str) -> dict[str, Any]:
        """Load results from a file."""

    def get_info(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "device": self.device,
        }
