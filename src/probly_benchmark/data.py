"""Collection of data loading functions."""

from __future__ import annotations

import ssl
from typing import Any

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.v2 as T

from probly_benchmark.paths import DATA_PATH

VAL_SPLIT = 0.2


def get_data_train(
    name: str,
    use_validation: bool = False,
    seed: int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Get data loaders for a dataset.

    Args:
        name: The name of the dataset.
        use_validation: Whether to use validation or test set. Defaults to False.
        seed: Seed for the random number generator. Defaults to None.
        **kwargs: Additional arguments passed to the data loader.

    Returns:
        A tuple of (train_loader, val_loader, test_loader). If use_validation is False, val_loader will be None.
    """
    name = name.lower()
    match name:
        case "cifar10":
            transforms_train = transforms_test = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            train = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms_train)
            test = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transforms_test)
        case "imagenet":
            transforms_train = transforms_test = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            train = torchvision.datasets.ImageNet(root=DATA_PATH, split="train", transform=transforms_train)
            test = torchvision.datasets.ImageNet(root=DATA_PATH, split="val", transform=transforms_test)
        case _:
            msg = f"Dataset {name} not recognized"
            raise ValueError(msg)

    if use_validation:
        generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        val_len = int(len(train) * VAL_SPLIT)
        train_len = len(train) - val_len
        train, val = torch.utils.data.random_split(train, [train_len, val_len], generator=generator)
        val_loader = torch.utils.data.DataLoader(val, **kwargs)
    else:
        val_loader = None
    train_loader = torch.utils.data.DataLoader(train, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, **kwargs)
    return train_loader, val_loader, test_loader


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
