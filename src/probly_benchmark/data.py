"""Collection of data loading functions."""

from __future__ import annotations

import ssl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Load MNIST dataset.

    Args:
        batch_size: Batch size.
    """
    ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]  # noqa: SLF001
    tf = transforms.ToTensor()
    train_data = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=tf)
    test_data = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
