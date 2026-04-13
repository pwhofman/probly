"""Collection of data loading functions."""

from __future__ import annotations

import ssl
from typing import Any

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.v2 as T
import webdataset as wds

from probly_benchmark.paths import DATA_PATH, IMAGENET_SHARD_PATH

VAL_SPLIT = 0.1
IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000


def _get_imagenet_sharded(
    use_validation: bool,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Get WebDataset-based loaders for sharded ImageNet.

    Args:
        use_validation: Whether to include a validation loader.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for CUDA transfers.
        persistent_workers: Whether DataLoader workers persist between epochs.
        prefetch_factor: Number of batches each worker prefetches.
        seed: Seed for deterministic shard and sample shuffling on the train
            loader. Val and test loaders do not shuffle, so the seed has no
            effect on them.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    # Decode directly to uint8 CHW torch tensors via torchvision.io (libjpeg-turbo);
    # resize on-tensor to avoid the slow PIL path.
    transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_shards = sorted(str(p) for p in IMAGENET_SHARD_PATH.glob("imagenet-train-*.tar"))
    val_shards = sorted(str(p) for p in IMAGENET_SHARD_PATH.glob("imagenet-val-*.tar"))

    if not train_shards:
        msg = f"No train shards found in {IMAGENET_SHARD_PATH}"
        raise FileNotFoundError(msg)
    if not val_shards:
        msg = f"No val shards found in {IMAGENET_SHARD_PATH}"
        raise FileNotFoundError(msg)

    def _make_loader(
        shards: list[str],
        shuffle_buf: int,
        num_samples: int,
        shardshuffle: bool = False,
        loader_seed: int | None = None,
    ) -> DataLoader:
        ds = wds.WebDataset(shards, shardshuffle=shardshuffle, seed=loader_seed)  # ty: ignore[unresolved-attribute]
        if shuffle_buf > 0:
            if loader_seed is not None:
                ds = ds.compose(wds.detshuffle(bufsize=shuffle_buf, seed=loader_seed))  # ty: ignore[unresolved-attribute]
            else:
                ds = ds.shuffle(shuffle_buf)
        ds = (
            ds.decode(wds.imagehandler("torchrgb8"))  # ty: ignore[unresolved-attribute]
            .to_tuple("jpg", "txt")
            .map_tuple(transform, int)
        )
        effective_num_workers = min(num_workers, len(shards))
        if effective_num_workers < num_workers:
            print(
                f"[imagenet_shards] Requested num_workers={num_workers} exceeds "
                f"shard count ({len(shards)}); capping to {effective_num_workers}."
            )
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": effective_num_workers,
            "pin_memory": pin_memory,
        }
        if effective_num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loader = wds.WebLoader(ds, **loader_kwargs)  # ty: ignore[unresolved-attribute]
        return loader.with_length(num_samples // batch_size)

    # split val shards deterministically to create a val and test set.
    n_val_shards = max(1, round(len(val_shards) * VAL_SPLIT))
    val_only_shards = val_shards[:n_val_shards]
    test_only_shards = val_shards[n_val_shards:]
    val_samples = IMAGENET_VAL_SIZE * n_val_shards // len(val_shards)
    test_samples = IMAGENET_VAL_SIZE - val_samples

    train_loader = _make_loader(
        train_shards,
        shuffle_buf=5000,
        num_samples=IMAGENET_TRAIN_SIZE,
        shardshuffle=True,
        loader_seed=seed,
    )
    val_loader = _make_loader(val_only_shards, shuffle_buf=0, num_samples=val_samples) if use_validation else None
    test_loader = _make_loader(test_only_shards, shuffle_buf=0, num_samples=test_samples)

    return train_loader, val_loader, test_loader


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
        case "imagenet_shards":
            return _get_imagenet_sharded(
                use_validation=use_validation,
                batch_size=kwargs["batch_size"],
                num_workers=kwargs.get("num_workers", 0),
                pin_memory=kwargs.get("pin_memory", False),
                persistent_workers=kwargs.get("persistent_workers", True),
                prefetch_factor=kwargs.get("prefetch_factor", 4),
                seed=seed,
            )
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
