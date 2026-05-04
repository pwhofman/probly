"""Collection of data loading functions."""

from __future__ import annotations

import copy
import itertools
import ssl
from typing import TYPE_CHECKING, Any, NamedTuple, cast
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.v2 as T
import webdataset as wds

from probly.datasets.torch import CIFAR10H
from probly_benchmark.paths import DATA_PATH, IMAGENET_SHARD_PATH, IMAGENET_TORCH_PATH

if TYPE_CHECKING:
    from pathlib import Path

    from omegaconf import DictConfig

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
            T.Resize((32, 32), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "cifar10h": T.Compose(
        [
            T.Resize((32, 32), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "imagenet": T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToImage(),
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
    train_common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }
    # Evaluation loaders don't need persistent workers; test is used only once
    # after potentially hours of training, so keeping workers alive wastes shared
    # memory and causes workers to be killed (SIGABRT) on long runs.
    eval_common: dict[str, Any] = {**train_common, "persistent_workers": False}

    train_loader = _make_imagenet_loader(
        alloc.train,
        alloc.train_samples,
        shuffle_buf=5000,
        shardshuffle=100,
        loader_seed=seed,
        **train_common,
    )
    val_loader = _make_imagenet_loader(alloc.val, alloc.val_samples, **eval_common) if alloc.val else None
    cal_loader = (
        _make_imagenet_loader(
            alloc.cal,
            alloc.cal_samples,
            shuffle_buf=5000,
            shardshuffle=100,
            loader_seed=seed,
            **eval_common,
        )
        if alloc.cal
        else None
    )
    test_loader = _make_imagenet_loader(alloc.test, alloc.test_samples, **eval_common)

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
    ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]  # noqa: SLF001
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
            # Val and test loaders don't shuffle; cal does (used for conformal/calibration).
            # None of them need persistent_workers — it wastes shared memory for loaders
            # used once and can cause workers to be killed (SIGABRT) on long runs.
            eval_kwargs: dict[str, Any] = {**kwargs, "shuffle": False, "persistent_workers": False}
            cal_kwargs: dict[str, Any] = {**kwargs, "shuffle": True, "persistent_workers": False}
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
                    val_loader = DataLoader(val, **eval_kwargs)
                if cal_split > 0:
                    cal.dataset = copy.copy(cal.dataset)
                    cal.dataset.transform = TRANSFORMS_TEST[name]  # ty: ignore[unresolved-attribute]
                    cal_loader = DataLoader(cal, **cal_kwargs)
            train_loader = DataLoader(train, **kwargs)
            test_loader = DataLoader(test, **eval_kwargs)
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
        case "imagenet_torch":
            train = torchvision.datasets.ImageNet(
                root=IMAGENET_TORCH_PATH,
                split="train",
                transform=TRANSFORMS_TEST["imagenet"],
            )
            train_loader = DataLoader(train, **kwargs)
            return DataLoaders(train_loader, None, None, None)  # ty: ignore
        case _:
            msg = f"Dataset {name} not recognized"
            raise ValueError(msg)


def build_laplace_fit_loader(
    train_loader: DataLoader,
    cfg: DictConfig,
    train_kwargs: dict[str, Any],
) -> DataLoader:
    """Build a DataLoader for ``BaseLaplace.fit``, honoring ``fit_batch_size`` and ``fit_subset``.

    laplace-torch's KFAC stores per-sample C-by-C Hessian square roots, so the fit step often
    needs a smaller batch than training. ``fit_subset`` (a fraction in ``(0, 1]``, or ``None``)
    trades Hessian-estimate variance for fit-time speed: the Hessian is an unbiased average
    across samples, so a uniform random subset gives the same estimator with higher variance.

    For ImageNet the source dataset is swapped from webdataset to ``imagenet_torch`` because
    laplace-torch requires ``len(loader.dataset)``.

    Args:
        train_loader: DataLoader used for the supervised fine-tune phase.
        cfg: Hydra config; ``cfg.dataset`` and ``cfg.seed`` are read.
        train_kwargs: ``train`` block from the method config; ``fit_batch_size`` (``int`` or
            ``None``; ``None`` means the train batch size, or 32 on ImageNet) and ``fit_subset``
            (``float`` in ``(0, 1]`` or ``None``) are read here.

    Returns:
        A DataLoader to pass to :meth:`laplace.baselaplace.BaseLaplace.fit`.

    Raises:
        ValueError: If ``fit_subset`` is not in ``(0, 1]``.
    """
    fit_subset = train_kwargs.get("fit_subset")
    if fit_subset is not None and not 0.0 < float(fit_subset) <= 1.0:
        msg = f"fit_subset must be in (0, 1] or None, got {fit_subset!r}"
        raise ValueError(msg)
    fit_batch_size_cfg = train_kwargs.get("fit_batch_size")

    if cfg.dataset == "imagenet":
        # KFAC with 1000 ImageNet classes stores CxC hessian square roots per sample;
        # training batch_size (2048) causes OOM, so default the fit batch to 32.
        fit_batch_size = int(fit_batch_size_cfg) if fit_batch_size_cfg is not None else 32
        print(
            f"[laplace-fit] cfg.dataset='imagenet' uses webdataset, which doesn't satisfy "
            f"laplace-torch's `len(loader.dataset)` requirement, hence switching to the "
            f"torchvision version (fit_batch_size={fit_batch_size})."
        )
        source_loader = get_data_train(
            "imagenet_torch",
            cfg.seed,
            val_split=cfg.val_split,
            cal_split=cfg.get("cal_split", 0.0),
            batch_size=fit_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
            prefetch_factor=cfg.get("prefetch_factor", 4),
            shuffle=True,
        ).train
    else:
        source_loader = train_loader
        fit_batch_size = int(fit_batch_size_cfg) if fit_batch_size_cfg is not None else source_loader.batch_size

    if fit_subset is None and fit_batch_size == source_loader.batch_size:
        return source_loader

    dataset = source_loader.dataset
    if fit_subset is not None:
        # Datasets fed to laplace-torch must be ``Sized`` (it calls ``len(loader.dataset)``);
        # ``torch.utils.data.Dataset`` itself doesn't promise ``__len__``, hence the cast.
        n_total = len(cast("Any", dataset))
        n_keep = max(1, round(float(fit_subset) * n_total))
        rng = torch.Generator().manual_seed(int(cfg.seed))
        indices = torch.randperm(n_total, generator=rng)[:n_keep].tolist()
        dataset = Subset(dataset, indices)
        print(f"[laplace-fit] fit_subset={fit_subset}: using {n_keep}/{n_total} samples")

    return DataLoader(
        dataset,
        batch_size=fit_batch_size,
        shuffle=True,
        num_workers=source_loader.num_workers,
        pin_memory=source_loader.pin_memory,
        persistent_workers=source_loader.num_workers > 0 and source_loader.persistent_workers,
    )


_GRAYSCALE_OOD_DATASETS = frozenset({"mnist", "fashion_mnist"})


class _HfOodDataset(torchvision.datasets.VisionDataset):
    """Torchvision-style wrapper for an HF-hosted OOD dataset.

    Bypasses ``datasets.load_dataset`` (which can misclassify tar layouts as
    WebDataset and which lazily streams xet-backed files row-by-row, causing
    pathological per-row network round-trips during iteration). Instead this
    downloads any archive files in the repo via ``hf_hub_download`` (full,
    once), extracts them to a local cache directory, and yields images by
    file path.

    Labels are returned as ``0`` because OOD evaluation does not consume them.
    """

    _CACHE_ROOT = DATA_PATH / "_hf_ood_cache"
    _IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    _ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".zip")

    def __init__(self, repo: str, transform: T.Compose) -> None:
        super().__init__(root="", transform=transform)
        extract_dir = self._ensure_local(repo)
        self._paths = sorted(p for p in extract_dir.rglob("*") if p.suffix.lower() in self._IMG_EXTS)
        if not self._paths:
            msg = f"No images found under {extract_dir} after extracting {repo}."
            raise RuntimeError(msg)
        self._transform = transform

    @classmethod
    def _ensure_local(cls, repo: str) -> Path:
        """Download and extract every archive file in ``repo`` once, locally."""
        import tarfile  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

        extract_dir = cls._CACHE_ROOT / repo.replace("/", "__")
        marker = extract_dir / ".extracted"
        if marker.exists():
            return extract_dir

        extract_dir.mkdir(parents=True, exist_ok=True)
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        archives = [f for f in files if f.endswith(cls._ARCHIVE_EXTS)]
        if not archives:
            msg = f"No tar/zip archives in {repo}; repo tree: {files}"
            raise RuntimeError(msg)

        for fname in archives:
            print(f"[_HfOodDataset] downloading {repo}:{fname}")
            local_path = hf_hub_download(repo, fname, repo_type="dataset")
            print(f"[_HfOodDataset] extracting {local_path} -> {extract_dir}")
            if fname.endswith(".zip"):
                with zipfile.ZipFile(local_path) as zf:
                    zf.extractall(extract_dir)  # noqa: S202
            else:
                with tarfile.open(local_path) as tf:
                    tf.extractall(extract_dir, filter="data")
        marker.touch()
        return extract_dir

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image  # noqa: PLC0415

        img = Image.open(self._paths[index]).convert("RGB")
        return self._transform(img), 0


def _get_test_dataset(name: str, transform: T.Compose) -> torchvision.datasets.VisionDataset:  # noqa: PLR0911, PLR0912
    """Return the test view of a map-style dataset with the given transform.

    Used by get_data_ood for both ID and OOD sides. Imagenet is not registered
    here; it goes through _allocate_imagenet_shards + _make_imagenet_loader.

    For grayscale OOD sources (see ``_GRAYSCALE_OOD_DATASETS``), an RGB
    conversion is prepended to the supplied transform so the ID-side normalize
    (which assumes 3 channels) applies without crashing.

    Args:
        name: Dataset name. One of ``cifar10``, ``cifar100``, ``svhn``,
            ``textures``, ``places365``, ``mnist``, ``fashion_mnist``,
            ``stl10``, ``eurosat``, ``sun397``, ``inaturalist``, ``ninco``,
            ``ssb_hard``.
        transform: Test transform to apply to all returned samples.

    Returns:
        A torch.utils.data.Dataset over the named dataset's test split. For
        ``eurosat`` and ``sun397``, which do not ship with a train/test split,
        the full dataset is returned. ``ninco`` and ``ssb_hard`` are loaded
        from HuggingFace mirrors of the original ImageNet near-OOD benchmarks.
    """
    ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]  # noqa: SLF001
    name = name.lower()
    if name in _GRAYSCALE_OOD_DATASETS:
        transform = T.Compose([T.RGB(), *transform.transforms])
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
        case "mnist":
            return torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
        case "fashion_mnist":
            return torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
        case "stl10":
            return torchvision.datasets.STL10(root=DATA_PATH, split="test", download=True, transform=transform)
        case "eurosat":
            return torchvision.datasets.EuroSAT(root=DATA_PATH, download=True, transform=transform)
        case "sun397":
            return torchvision.datasets.SUN397(root=DATA_PATH, download=True, transform=transform)
        case "inaturalist":
            return torchvision.datasets.INaturalist(
                root=DATA_PATH, version="2021_valid", download=True, transform=transform
            )
        case "ninco":
            return _HfOodDataset("Rxzh/NINCO", transform)
        case "ssb_hard":
            return _HfOodDataset("torch-uncertainty/SSB_hard", transform)
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


def get_data_first_order(
    name: str,
    **kwargs: Any,  # noqa: ANN401
) -> DataLoader:
    """Get the test loader for first-order data comparison.

    Returns the test split for comparing model predictions against
    ground-truth human label distributions (e.g. CIFAR-10H).
    ``val_split`` and ``cal_split`` are accepted for API consistency and
    future dataset support but are unused for CIFAR-10.

    Args:
        name: Dataset name. Currently only ``"cifar10"`` is supported.
        seed: Random seed. Unused for CIFAR-10 (no random splitting).
        val_split: Validation split fraction. Unused for CIFAR-10.
        cal_split: Calibration split fraction. Unused for CIFAR-10.
        **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.

    Returns:
        DataLoader over the test set.

    Raises:
        ValueError: If ``name`` is not a recognized dataset.
    """
    name = name.lower()
    match name:
        case "cifar10h":
            data = CIFAR10H(root=DATA_PATH, download=True, transform=TRANSFORMS_TEST["cifar10"])
            return DataLoader(data, **kwargs)
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
        name_ood: Out-of-distribution dataset name. One of cifar10, cifar100,
            svhn, textures, places365, mnist, fashion_mnist, stl10, eurosat,
            sun397, inaturalist, ninco, ssb_hard.
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


def get_data_al(
    name: str,
    seed: int,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load a dataset as torch tensors for active learning.

    Args:
        name: Dataset name (cifar10, fashion_mnist, openml).
        seed: Random seed for train/test splitting (OpenML only).
        **kwargs: Extra arguments. ``openml_id`` is required for OpenML datasets.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test, num_classes, in_features).
    """
    name = name.lower()
    match name:
        case "cifar10":
            transform = TRANSFORMS_TEST["cifar10"]
            train_ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
            x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
            y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
            x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
            y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
            num_classes = 10
            in_features = x_train.shape[1]
        case "fashion_mnist":
            transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
            train_ds = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True, transform=transform)
            x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
            y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)
            x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
            y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)
            num_classes = 10
            in_features = x_train.shape[1]
        case "openml":
            from sklearn.datasets import fetch_openml  # noqa: PLC0415
            from sklearn.model_selection import train_test_split  # noqa: PLC0415
            from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: PLC0415

            openml_id = kwargs.get("openml_id")
            if openml_id is None:
                msg = "openml_id is required for OpenML datasets."
                raise ValueError(msg)
            dataset = fetch_openml(data_id=openml_id, as_frame=False, parser="auto")
            x = dataset.data.astype(np.float32)
            le = LabelEncoder()
            y = le.fit_transform(dataset.target)
            x_np_train, x_np_test, y_np_train, y_np_test = train_test_split(
                x, y, test_size=0.2, random_state=seed, stratify=y
            )
            scaler = StandardScaler()
            x_train = torch.from_numpy(scaler.fit_transform(x_np_train).astype(np.float32))
            x_test = torch.from_numpy(scaler.transform(x_np_test).astype(np.float32))
            y_train = torch.from_numpy(y_np_train.astype(np.int64))
            y_test = torch.from_numpy(y_np_test.astype(np.int64))
            num_classes = len(le.classes_)
            in_features = x_train.shape[1]
        case _:
            msg = f"Dataset {name} not recognized for active learning."
            raise ValueError(msg)
    return x_train, y_train, x_test, y_test, num_classes, in_features
