"""
Abstract base class: Defines the interface that all data generators must implement.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseDataGenerator(ABC):
    """Base class for all data generators"""

    @abstractmethod
    def __init__(
        self,
        model: Any,
        dataset: Any,
        batch_size: int = 32,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize data generator"""
        pass

    @abstractmethod
    def generate_distributions(self) -> Dict[str, Any]:
        """
        Generate probability distribution data

        Returns:
            Dictionary containing probability distributions
        """
        pass

    @abstractmethod
    def save_distributions(self, path: str) -> None:
        """
        Save the generated data

        Args:
            path: Save path
        """
        pass

    @abstractmethod
    def load_distributions(self, path: str) -> Dict[str, Any]:
        """
        Load saved data

        Args:
            path: File path

        Returns:
            Loaded data dictionary
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get generator information

        Returns:
            Dictionary containing information
        """
        return {
            "framework": "unknown",
            "batch_size": getattr(self, 'batch_size', None),
            "device": getattr(self, 'device', None)
        }
