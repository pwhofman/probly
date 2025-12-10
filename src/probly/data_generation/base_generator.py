from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDataGenerator(ABC):
    """
    Base class for data generators used in our project.
    """

    def __init__(self, model, dataset, batch_size=32, device=None):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """Run the model on the dataset and collect statistics."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save generated results to a file."""
        pass

    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """Load results from a file."""
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "device": self.device
        }
