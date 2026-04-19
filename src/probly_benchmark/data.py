"""Collection of data loading functions."""

from __future__ import annotations

import copy
import itertools
import ssl
from typing import Any, NamedTuple
import warnings

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.v2 as T
import webdataset as wds

from probly_benchmark.paths import DATA_PATH, IMAGENET_SHARD_PATH

# Ignore a warning from WebDataset about the use of length
warnings.filterwarnings("ignore", message=".*with_length\\(\\).*")


class DataLoaders(NamedTuple):
    """Data loaders for a dataset."""

    train: DataLoader
    validation: DataLoader | None
    calibration: DataLoader | None
    test: DataLoader


TRANSFORMS_TEST = {
    "cifar10": T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "imagenet": T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
}


IMAGENET_TRAIN_SIZE = 1_281_167
IMAGENET_VAL_SIZE = 50_000


class _ImagenetShards(NamedTuple):
    """Imagenet shard allocation: shard paths plus sample counts per split."""

    train: list[str]
    val: list[str]
    cal: list[str]
    test: list[str]
    train_samples: int
    val_samples: int
    cal_samples: int
    test_samples: int


def _allocate_imagenet_shards(val_split: float, cal_split: float) -> _ImagenetShards:
    """Allocate imagenet shards across train, val, cal, and test deterministically.

    Train shards always go entirely to train. Validation and calibration are
    carved from the val shards at shard granularity (whole shards, sorted
    order). When a requested fraction is smaller than one shard, one shard is
    used and a deviation message is printed. Allocation depends only on the
    two fractions and the sorted shard order, so training and selective
    prediction get identical test partitions for the same (val_split, cal_split).

    Args:
        val_split: Fraction of val shards to allocate to validation. 0 disables.
        cal_split: Fraction of val shards to allocate to calibration. 0 disables.

    Returns:
        An _ImagenetShards NamedTuple. Val and cal lists are empty when their
        split is 0.
    """
    train_shards = sorted(str(p) for p in IMAGENET_SHARD_PATH.glob("imagenet-train-*.tar"))
    val_shards = sorted(str(p) for p in IMAGENET_SHARD_PATH.glob("imagenet-val-*.tar"))

    if not train_shards:
        msg = f"No train shards found in {IMAGENET_SHARD_PATH}"
        raise FileNotFoundError(msg)
    if not val_shards:
        msg = f"No val shards found in {IMAGENET_SHARD_PATH}"
        raise FileNotFoundError(msg)

    def _allocate(split: float, total: int, name: str) -> int:
        if split <= 0:
            return 0
        exact = total * split
        if exact < 1:
            print(
                f"[imagenet_shards] Requested {name}_split={split} corresponds to "
                f"{exact:.2f} shards (<1); using 1 shard instead "
                f"(~{IMAGENET_VAL_SIZE // total} samples). Effective fraction: "
                f"{1 / total:.4f}."
            )
            return 1
        return round(exact)

    n_val = _allocate(val_split, len(val_shards), "val")
    n_cal = _allocate(cal_split, len(val_shards), "cal")

    if n_val + n_cal >= len(val_shards):
        msg = f"val_split + cal_split allocates {n_val + n_cal} of {len(val_shards)} val shards, leaving none for test."
        raise ValueError(msg)

    val_samples = IMAGENET_VAL_SIZE * n_val // len(val_shards)
    cal_samples = IMAGENET_VAL_SIZE * n_cal // len(val_shards)
    test_samples = IMAGENET_VAL_SIZE - val_samples - cal_samples

    return _ImagenetShards(
        train=train_shards,
        val=val_shards[:n_val],
        cal=val_shards[n_val : n_val + n_cal],
        test=val_shards[n_val + n_cal :],
        train_samples=IMAGENET_TRAIN_SIZE,
        val_samples=val_samples,
        cal_samples=cal_samples,
        test_samples=test_samples,
    )


def _make_imagenet_loader(
    shards: list[str],
    num_samples: int,
    *,
    batch_size: int,
    shuffle_buf: int = 0,
    shardshuffle: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    loader_seed: int | None = None,
) -> DataLoader:
    """Build a single WebDataset-based loader for imagenet shards.

    Decodes directly to uint8 CHW torch tensors via torchvision.io
    (libjpeg-turbo); resizes on-tensor to avoid the slow PIL path.

    Args:
        shards: Shard file paths to load from.
        num_samples: Total samples across the given shards (used for length).
        batch_size: Batch size.
        shuffle_buf: Sample-shuffle buffer size. 0 disables sample shuffling.
        shardshuffle: Shard-shuffle buffer size. 0 disables shard shuffling.
        num_workers: Number of data loading workers. Capped to len(shards).
        pin_memory: Whether to pin memory for CUDA transfers.
        persistent_workers: Whether DataLoader workers persist between epochs.
        prefetch_factor: Number of batches each worker prefetches.
        loader_seed: Seed for deterministic shuffling. None gives a fresh
            (non-reproducible) seed.

    Returns:
        A WebLoader with length set to num_samples // batch_size.
    """
    ds = wds.WebDataset(shards, shardshuffle=shardshuffle, seed=loader_seed)  # ty: ignore[unresolved-attribute]
    if shuffle_buf > 0:
        if loader_seed is not None:
            ds = ds.compose(wds.detshuffle(bufsize=shuffle_buf, seed=loader_seed))  # ty: ignore[unresolved-attribute]
        else:
            ds = ds.shuffle(shuffle_buf)
    ds = (
        ds.decode(wds.imagehandler("torchrgb8"))  # ty: ignore[unresolved-attribute]
        .to_tuple("jpg", "txt")
        .map_tuple(TRANSFORMS_TEST["imagenet"], int)
        .compose(lambda src: itertools.islice(src, num_samples))
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


def _get_imagenet_sharded(
    val_split: float,
    cal_split: float,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int | None = None,
) -> DataLoaders:
    """Get WebDataset-based loaders for sharded ImageNet.

    Validation and calibration sets are carved from the val shards at shard
    granularity (whole shards, deterministic on sorted shard order). When the
    requested fraction is smaller than one shard, one shard is used and a
    deviation message is printed.

    Args:
        val_split: Fraction of val shards to allocate to validation. 0 disables.
        cal_split: Fraction of val shards to allocate to calibration. 0 disables.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for CUDA transfers.
        persistent_workers: Whether DataLoader workers persist between epochs.
        prefetch_factor: Number of batches each worker prefetches.
        seed: Seed for deterministic shard and sample shuffling on the train
            loader. Val, calibration, and test loaders do not shuffle, so the
            seed has no effect on them.

    Returns:
        A DataLoaders NamedTuple with train, validation, calibration, and test
        loaders. Validation and calibration are None when their split is 0.
    """
    alloc = _allocate_imagenet_shards(val_split, cal_split)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }

    train_loader = _make_imagenet_loader(
        alloc.train,
        alloc.train_samples,
        shuffle_buf=5000,
        shardshuffle=100,
        loader_seed=seed,
        **common,
    )
    val_loader = _make_imagenet_loader(alloc.val, alloc.val_samples, **common) if alloc.val else None
    cal_loader = _make_imagenet_loader(alloc.cal, alloc.cal_samples, **common) if alloc.cal else None
    test_loader = _make_imagenet_loader(alloc.test, alloc.test_samples, **common)

    return DataLoaders(train_loader, val_loader, cal_loader, test_loader)


def get_data_train(
    name: str,
    seed: int,
    val_split: float = 0.0,
    cal_split: float = 0.0,
    **kwargs: Any,  # noqa: ANN401
) -> DataLoaders:
    """Get data loaders for a dataset.

    Args:
        name: The name of the dataset.
        seed: Seed for the random number generator.
        val_split: Split for the validation set.
        cal_split: Split for the calibration set.
        **kwargs: Additional arguments passed to the data loader.

    Returns:
        A NamedTuple of (train_loader, val_loader, cal_loader, test_loader),
            where val_loader and cal_loader may be None if their splits are 0.
    """
    name = name.lower()
    match name:
        case "cifar10":
            transforms_train = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            train = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transforms_train)
            test = torchvision.datasets.CIFAR10(
                root=DATA_PATH, train=False, download=True, transform=TRANSFORMS_TEST[name]
            )
            val_loader = None
            cal_loader = None
            if val_split > 0 or cal_split > 0:
                rng = torch.Generator().manual_seed(seed)
                val_len = int(len(train) * val_split)
                cal_len = int(len(train) * cal_split)
                train_len = len(train) - val_len - cal_len
                train, val, cal = random_split(train, [train_len, val_len, cal_len], generator=rng)
                if val_split > 0:
                    val.dataset = copy.copy(val.dataset)
                    val.dataset.transform = TRANSFORMS_TEST[name]  # ty: ignore[unresolved-attribute]
                    val_loader = DataLoader(val, **kwargs)
                if cal_split > 0:
                    cal.dataset = copy.copy(cal.dataset)
                    cal.dataset.transform = TRANSFORMS_TEST[name]  # ty: ignore[unresolved-attribute]
                    cal_loader = DataLoader(cal, **kwargs)
            train_loader = DataLoader(train, **kwargs)
            test_loader = DataLoader(test, **kwargs)
            return DataLoaders(train_loader, val_loader, cal_loader, test_loader)
        case "imagenet":
            return _get_imagenet_sharded(
                val_split=val_split,
                cal_split=cal_split,
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


def _get_test_dataset(name: str, transform: T.Compose) -> torchvision.datasets.VisionDataset:
    """Return the test view of a map-style dataset with the given transform.

    Used by get_data_ood for both ID and OOD sides. Imagenet is not registered
    here; it goes through _allocate_imagenet_shards + _make_imagenet_loader.

    Args:
        name: Dataset name. One of cifar10, cifar100, svhn, textures, places365.
        transform: Test transform to apply to all returned samples.

    Returns:
        A torch.utils.data.Dataset over the named dataset's test split.
    """
    name = name.lower()
    match name:
        case "cifar10":
            return torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
        case "cifar100":
            return torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=transform)
        case "svhn":
            return torchvision.datasets.SVHN(root=DATA_PATH, split="test", download=True, transform=transform)
        case "textures":
            return torchvision.datasets.DTD(root=DATA_PATH, split="test", download=True, transform=transform)
        case "places365":
            return torchvision.datasets.Places365(
                root=DATA_PATH, split="val", small=True, download=True, transform=transform
            )
        case _:
            msg = f"Dataset {name} not recognized for test loading"
            raise ValueError(msg)


def get_data_selective_prediction(
    name: str,
    seed: int,  # noqa: ARG001
    val_split: float = 0.0,
    cal_split: float = 0.0,
    **kwargs: Any,  # noqa: ANN401
) -> DataLoader:
    """Get the test loader to do the selective prediction task."""
    name = name.lower()
    match name:
        case "cifar10":
            test = torchvision.datasets.CIFAR10(
                root=DATA_PATH, train=False, download=True, transform=TRANSFORMS_TEST[name]
            )
            return DataLoader(test, **kwargs)
        case "imagenet":
            alloc = _allocate_imagenet_shards(val_split, cal_split)
            return _make_imagenet_loader(
                alloc.test,
                alloc.test_samples,
                batch_size=kwargs["batch_size"],
                num_workers=kwargs.get("num_workers", 0),
                pin_memory=kwargs.get("pin_memory", False),
                persistent_workers=kwargs.get("persistent_workers", True),
                prefetch_factor=kwargs.get("prefetch_factor", 4),
            )
        case _:
            msg = f"Dataset {name} not recognized"
            raise ValueError(msg)


def get_data_ood(
    name_id: str,
    name_ood: str,
    seed: int,
    val_split: float = 0.0,
    cal_split: float = 0.0,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[DataLoader, DataLoader]:
    """Get ID and OOD test loaders for an OOD detection experiment.

    Both loaders are leakage-safe (no train/val/cal samples), produce exactly
    the same number of instances (min of the two pool sizes), and apply the
    ID dataset's test transform on both sides.

    For name_id == "imagenet" the islice truncation requires single-process
    iteration; num_workers is forced to 0 on the ID side and a deviation
    message is printed if a higher value was passed.

    Args:
        name_id: In-distribution dataset name. cifar10 or imagenet.
        name_ood: Out-of-distribution dataset name. One of cifar100, svhn,
            textures, places365.
        seed: Seed driving the random subset selection on both sides.
        val_split: Fraction allocated to validation during training. Used to
            ensure ID test excludes those samples (no-op for cifar10).
        cal_split: Fraction allocated to calibration during training. Used to
            ensure ID test excludes those samples (no-op for cifar10).
        **kwargs: DataLoader kwargs (batch_size, num_workers, pin_memory,
            persistent_workers, prefetch_factor, ...).

    Returns:
        A tuple (id_loader, ood_loader) of equal-size test loaders.
    """
    name_id = name_id.lower()
    name_ood = name_ood.lower()
    transform = TRANSFORMS_TEST[name_id]
    ood_ds = _get_test_dataset(name_ood, transform)

    match name_id:
        case "cifar10":
            id_ds = _get_test_dataset(name_id, transform)
            n_id = len(id_ds)
            target = min(n_id, len(ood_ds))
            rng = torch.Generator().manual_seed(seed)
            id_idx = torch.randperm(n_id, generator=rng)[:target].tolist()
            ood_idx = torch.randperm(len(ood_ds), generator=rng)[:target].tolist()
            id_loader = DataLoader(Subset(id_ds, id_idx), **kwargs)
            ood_loader = DataLoader(Subset(ood_ds, ood_idx), **kwargs)
        case "imagenet":
            alloc = _allocate_imagenet_shards(val_split, cal_split)
            n_id = alloc.test_samples
            target = min(n_id, len(ood_ds))
            requested_workers = kwargs.get("num_workers", 0)
            if requested_workers > 0:
                print(
                    f"[get_data_ood] Capping num_workers from {requested_workers} "
                    f"to 0 for ID imagenet path (islice truncation requires "
                    f"single-process iteration to yield exactly {target} samples)."
                )
            id_loader = _make_imagenet_loader(
                alloc.test,
                num_samples=target,
                batch_size=kwargs["batch_size"],
                num_workers=0,
                pin_memory=kwargs.get("pin_memory", False),
                persistent_workers=kwargs.get("persistent_workers", True),
                prefetch_factor=kwargs.get("prefetch_factor", 4),
                shuffle_buf=target,
                shardshuffle=100,
                loader_seed=seed,
            )
            rng = torch.Generator().manual_seed(seed)
            ood_idx = torch.randperm(len(ood_ds), generator=rng)[:target].tolist()
            ood_loader = DataLoader(Subset(ood_ds, ood_idx), **kwargs)
        case _:
            msg = f"Dataset {name_id} not recognized as ID dataset"
            raise ValueError(msg)

    print(
        f"[get_data_ood] ID={name_id}: {n_id} samples, OOD={name_ood}: {len(ood_ds)} samples. Using {target} for both."
    )
    return id_loader, ood_loader


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
