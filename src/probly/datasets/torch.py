"""Collection of dataset classes for loading data from different datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ClassVar


class CIFAR10H(torchvision.datasets.CIFAR10):
    """A Dataset class for the CIFAR10H dataset introduced in :cite:`petersonHumanUncertainty2019`.

    The dataset can be found at https://github.com/jcpeterson/cifar-10h.
    """

    counts: torch.Tensor
    """Tensor containing counts."""

    targets: torch.Tensor
    """Tensor of size (n_instances, n_classes), first-order distribution."""

    def __init__(
        self, root: str | Path, transform: Callable[..., Any] | None = None, *, download: bool = False
    ) -> None:
        """Initialize an instance of the CIFAR10H class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            download: Whether to download the CIFAR10 dataset or not.
        """
        super().__init__(root, train=False, transform=transform, download=download)
        first_order_path = Path(self.root) / "cifar-10h-master" / "data" / "cifar10h-counts.npy"
        self.counts = np.load(first_order_path)
        self.counts = torch.tensor(self.counts, dtype=torch.float32)
        self.targets = self.counts / self.counts.sum(dim=1, keepdim=True)


class ImageNetReaL(torchvision.datasets.ImageNet):
    """A Dataset class for the ImageNet ReaL dataset introduced in :cite:`beyerDoneImageNet2020`.

    This dataset is a re-labeled version of the ImageNet validation set, where each image can belong
    to multiple classes resulting in a distribution over classes.
    The ImageNet dataset needs to be downloaded from https://www.image-net.org and the first order labels can be
    downloaded from https://github.com/google-research/reassessed-imagenet.
    """

    dists: list
    """List of distributions over target classes."""

    def __init__(self, root: str | Path, transform: Callable[..., Any] | None = None) -> None:
        """Initialize an instance of the ImageNetReaL class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
        """
        super().__init__(root=root, split="val", transform=transform)
        root = Path(root).expanduser()
        with (Path(root).expanduser() / "reassessed-imagenet-master/real.json").open() as f:
            real = json.load(f)
        real_labels = {f"ILSVRC2012_val_{(i + 1):08d}.JPEG": labels for i, labels in enumerate(real)}
        self.dists = []
        for img, _ in self.samples:
            labels = real_labels[img.split("/")[-1]]
            if labels:
                dist = torch.zeros(len(self.classes))
                dist[labels] = 1
                dist = dist / dist.sum()
            else:
                dist = torch.ones(len(self.classes)) / len(self.classes)
            self.dists.append(dist)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the item at the specified index.

        Args:
            index: Index.

        Returns:
            The tuple (sample, dist) where dist is a distribution over target classes.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        dist = self.dists[index]
        return sample, dist


class DCICDataset(torch.utils.data.Dataset):
    """A Dataset base class for the DCICDatasets introduced in :cite:`schmarjeIsOne2022`.

    These datasets can be found at https://zenodo.org/records/7180818.
    """

    root: Path
    """Root directory of the dataset."""

    transform: Callable[..., Any] | None
    """Transform to apply to the data."""

    image_labels: dict[str, list[int]]
    """Dictionary of image labels grouped by image."""

    image_paths: list[str]
    """List of image paths."""

    label_mappings: dict[int, int]
    """Mapping of labels to indices."""

    num_classes: int
    """Number of classes."""

    targets: list[torch.Tensor]
    """List of labels."""

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the DCICDataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        root = Path(root).expanduser()
        with (Path(root).expanduser() / "annotations.json").open() as f:
            annotations = json.load(f)

        self.root = root.parent
        self.transform = transform
        self.image_labels = {}

        for entry in annotations:
            for annotation in entry["annotations"]:
                img_path = annotation["image_path"]
                label = annotation["class_label"]

                if img_path not in self.image_labels:
                    self.image_labels[img_path] = []

                self.image_labels[img_path].append(label)

        self.image_paths = list(self.image_labels.keys())
        unique_labels = {label for labels in self.image_labels.values() for label in labels}
        self.label_mappings = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len({label for labels in self.image_labels.values() for label in labels})
        self.targets = []
        for img_path in self.image_paths:
            labels = self.image_labels[img_path]
            label_indices = [self.label_mappings[label] for label in labels]
            dist = torch.bincount(torch.tensor(label_indices), minlength=self.num_classes).float()
            dist /= dist.sum()
            if first_order:
                self.targets.append(dist)
            else:
                self.targets.append(torch.multinomial(dist, 1).squeeze())

    def __len__(self) -> int:
        """Return the number of instances in the dataset.

        Returns:
            The number of instances in the dataset.

        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        """Returned indexed item in the dataset.

        Args:
            index: Index within the dataset.

        Returns:
            The (image, target) tuple within the dataset.
        """
        image_path = Path(self.root) / self.image_paths[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.targets[index]
        return image, target


class Benthic(DCICDataset):
    """Implementation of the Benthic dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Benthic dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Benthic", transform, first_order=first_order)


class CIFAR10HDCIC(DCICDataset):
    """Implementation of the DCIC version of the CIFAR10H dataset.

    Targets and labels are the same as in the original CIFAR10H dataset.
    This variant uses the DCIC file structure and predefined folds.
    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the DCIC-version of the CIFAR10H dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "CIFAR10H", transform, first_order=first_order)


class MiceBone(DCICDataset):
    """Implementation of the MiceBone dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the MiceBone dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "MiceBone", transform, first_order=first_order)


class Pig(DCICDataset):
    """Implementation of the Pig dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Pig dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Pig", transform, first_order=first_order)


class Plankton(DCICDataset):
    """Implementation of the Plankton dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Plankton dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Plankton", transform, first_order=first_order)


class QualityMRI(DCICDataset):
    """Implementation of the QualityMRI dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the QualityMRI dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "QualityMRI", transform, first_order=first_order)


class Synthetic(DCICDataset):
    """Implementation of the Synthetic dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Synthetic dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Synthetic", transform, first_order=first_order)


class Turkey(DCICDataset):
    """Implementation of the Turkey dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Turkey dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Turkey", transform, first_order=first_order)


class Treeversity1(DCICDataset):
    """Implementation of the Treeversity#1 dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Treeversity#1 dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Treeversity#1", transform, first_order=first_order)


class Treeversity6(DCICDataset):
    """Implementation of the Treeversity#6 dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Callable[..., Any] | None = None,
        *,
        first_order: bool = True,
    ) -> None:
        """Initialize an instance of the Treeversity#6 dataset class.

        Args:
            root: Root directory of the dataset.
            transform: Optional transform to apply to the data.
            first_order: Whether to use first order data or class labels. Defaults to True.
        """
        super().__init__(Path(root) / "Treeversity#6", transform, first_order=first_order)


class CIFAR10C(torchvision.datasets.VisionDataset):
    """A Dataset class for the CIFAR-10-C corruption benchmark introduced in :cite:`hendrycksBenchmarkingNeural2019`.

    One instance holds the 10,000 CIFAR-10 test images for a single ``corruption`` type at a single
    ``severity`` level (1-5), with hard integer labels. The data can be found at
    https://zenodo.org/records/2535967 and fetched with ``download=True`` (a single ~2.9 GB
    ``CIFAR-10-C.tar`` that covers all corruption types).
    """

    base_folder = "CIFAR-10-C"
    url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"
    tar_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    corruptions: tuple[str, ...] = (
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "speckle_noise",
        "gaussian_blur",
        "spatter",
        "saturate",
    )
    """The 19 corruption types shipped with CIFAR-10-C (15 main + 4 extra)."""

    data: np.ndarray
    """Array of shape (10000, 32, 32, 3), uint8, for the selected corruption and severity."""

    targets: list[int]
    """Hard integer class labels, one per image."""

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *,
        download: bool = False,
    ) -> None:
        """Initialize an instance of the CIFAR10C class.

        Args:
            root: Root directory containing (or to download into) the ``CIFAR-10-C`` folder.
            corruption: Corruption type; must be one of ``CIFAR10C.corruptions``.
            severity: Corruption severity in 1..5.
            transform: Optional transform to apply to the image.
            target_transform: Optional transform to apply to the integer label.
            download: Whether to download the CIFAR-10-C tar from Zenodo if missing.

        Raises:
            ValueError: If ``corruption`` is unknown or ``severity`` is not in 1..5.
            RuntimeError: If the data is missing and ``download`` is False.
        """
        super().__init__(str(root), transform=transform, target_transform=target_transform)
        if corruption not in self.corruptions:
            msg = f"Unknown corruption {corruption!r}. Valid options: {', '.join(self.corruptions)}."
            raise ValueError(msg)
        if not 1 <= severity <= 5:
            msg = f"severity must be in 1..5, got {severity}."
            raise ValueError(msg)
        self.corruption = corruption
        self.severity = severity

        folder = Path(self.root) / self.base_folder
        data_path = folder / f"{corruption}.npy"
        labels_path = folder / "labels.npy"

        if not (data_path.exists() and labels_path.exists()):
            if not download:
                msg = "Dataset not found. Use download=True to download it."
                raise RuntimeError(msg)
            download_and_extract_archive(self.url, str(self.root), filename=self.filename, md5=self.tar_md5)

        all_data = np.load(data_path, mmap_mode="r")  # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path)  # (50000,)
        n = len(all_data) // 5  # images per severity (10000 for the real dataset)
        sl = slice((severity - 1) * n, severity * n)
        self.data = np.ascontiguousarray(all_data[sl])  # materialize only this severity
        self.targets = all_labels[sl].tolist()

    def __len__(self) -> int:
        """Return the number of images (10000 for the real dataset).

        Returns:
            The number of images in this corruption/severity slice.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Get the (image, label) pair at the given index.

        Args:
            index: Index within the dataset.

        Returns:
            The ``(image, label)`` tuple, with ``transform``/``target_transform`` applied.
        """
        img = Image.fromarray(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class MedMNISTC(torchvision.datasets.VisionDataset):
    """A Dataset class for the MedMNIST-C corruption benchmark introduced in :cite:`disalvoMedMNISTC2024`.

    One instance holds one MedMNIST test set for a single dataset (``dataset``, e.g. ``"dermamnist"``),
    ``corruption`` type, and ``severity`` level (1-5), with hard integer labels. The release ships one
    ``.npz`` per ``(dataset, corruption)`` at 224x224 (https://zenodo.org/records/11471504), stacking all
    five severities in ``test_images`` as ``(5*N, H, W[, C])`` uint8 with the same severity-major layout
    as CIFAR-10-C, and can be fetched with ``download=True``. Loading is resolution-agnostic, so pass a
    ``transform`` (e.g. ``Resize``) to reach a different resolution. ChestMNIST is excluded because it is
    the only multi-label MedMNIST2D task, whereas this loader returns hard integer labels.
    """

    base_folder = "medmnist_c"
    md5s: ClassVar[dict[str, str]] = {
        "pathmnist": "bf62498906ec0383c3ec5ff12ac70c00",
        "bloodmnist": "daba10a010064a9e38f0d09b498bcf18",
        "dermamnist": "19c88c74c104655d5f668e158e56451d",
        "retinamnist": "80a2fa4c9b7fa2176606be825dfdef6e",
        "octmnist": "40389bc54256edecd09ee4bd028c7e6a",
        "breastmnist": "c755e51825c074706524ea6b2c77a10b",
        "pneumoniamnist": "c499a47a64a000b579b23920bf80a95a",
        "organamnist": "ff4ad0f53934ddf6a2330065d581990c",
        "organcmnist": "2a649c94473ab9126130af8205d6c89b",
        "organsmnist": "1dbf4d814725adc307ba9ffe5edf061a",
    }
    """Per-dataset md5 of the released archive at https://zenodo.org/records/11471504 (10 of the 11
    supported datasets; TissueMNIST-C is registry-only and must be generated locally)."""

    corruptions: ClassVar[dict[str, tuple[str, ...]]] = {
        "pathmnist": (
            "pixelate",
            "jpeg_compression",
            "defocus_blur",
            "motion_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "saturate",
            "stain_deposit",
            "bubble",
        ),
        "bloodmnist": (
            "pixelate",
            "jpeg_compression",
            "defocus_blur",
            "motion_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "saturate",
            "stain_deposit",
            "bubble",
        ),
        "dermamnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "impulse_noise",
            "shot_noise",
            "defocus_blur",
            "motion_blur",
            "zoom_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "black_corner",
            "characters",
        ),
        "retinamnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "defocus_blur",
            "motion_blur",
            "brightness_down",
            "contrast_down",
        ),
        "tissuemnist": (
            "pixelate",
            "jpeg_compression",
            "impulse_noise",
            "gaussian_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
        ),
        "octmnist": (
            "pixelate",
            "jpeg_compression",
            "speckle_noise",
            "defocus_blur",
            "motion_blur",
            "contrast_down",
        ),
        "breastmnist": (
            "pixelate",
            "jpeg_compression",
            "speckle_noise",
            "motion_blur",
            "brightness_up",
            "brightness_down",
            "contrast_down",
        ),
        "pneumoniamnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "impulse_noise",
            "shot_noise",
            "gaussian_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "gamma_corr_up",
            "gamma_corr_down",
        ),
        "organamnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "impulse_noise",
            "shot_noise",
            "gaussian_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "gamma_corr_up",
            "gamma_corr_down",
        ),
        "organcmnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "impulse_noise",
            "shot_noise",
            "gaussian_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "gamma_corr_up",
            "gamma_corr_down",
        ),
        "organsmnist": (
            "pixelate",
            "jpeg_compression",
            "gaussian_noise",
            "speckle_noise",
            "impulse_noise",
            "shot_noise",
            "gaussian_blur",
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "contrast_down",
            "gamma_corr_up",
            "gamma_corr_down",
        ),
    }
    """Per-dataset corruption types shipped with MedMNIST-C (arXiv:2406.17536, Table 1)."""

    data: np.ndarray
    """Array of shape (N, H, W) or (N, H, W, C), uint8, for the selected dataset/corruption/severity."""

    targets: list[int]
    """Hard integer class labels, one per image."""

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        corruption: str,
        severity: int,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *,
        download: bool = False,
    ) -> None:
        """Initialize an instance of the MedMNISTC class.

        Args:
            root: Root directory containing (or to download into) the ``medmnist_c`` folder.
            dataset: Which MedMNIST dataset to load (its MedMNIST "flag"); must be one of ``MedMNISTC.corruptions``.
            corruption: Corruption type; must be one of ``MedMNISTC.corruptions[dataset]``.
            severity: Corruption severity in 1..5.
            transform: Optional transform to apply to the image.
            target_transform: Optional transform to apply to the integer label.
            download: Whether to download the dataset's archive from Zenodo if missing. Not every dataset
                is published on Zenodo (e.g. ``"tissuemnist"`` must be generated locally).

        Raises:
            ValueError: If ``dataset`` or ``corruption`` is unknown or ``severity`` is not in 1..5.
            RuntimeError: If the data is missing and ``download`` is False, or if ``download`` is
                requested for a dataset that is not available on Zenodo.
        """
        super().__init__(str(root), transform=transform, target_transform=target_transform)
        if dataset not in self.corruptions:
            msg = f"Unknown MedMNIST dataset {dataset!r}. Valid options: {', '.join(self.corruptions)}."
            raise ValueError(msg)
        if corruption not in self.corruptions[dataset]:
            valid = ", ".join(self.corruptions[dataset])
            msg = f"Unknown corruption {corruption!r} for {dataset!r}. Valid options: {valid}."
            raise ValueError(msg)
        if not 1 <= severity <= 5:
            msg = f"severity must be in 1..5, got {severity}."
            raise ValueError(msg)
        self.dataset = dataset
        self.corruption = corruption
        self.severity = severity

        folder = Path(self.root) / self.base_folder
        npz_path = folder / dataset / f"{corruption}.npz"

        if not npz_path.exists():
            if not download:
                msg = "Dataset not found. Use download=True to download it."
                raise RuntimeError(msg)
            if dataset not in self.md5s:
                msg = f"{dataset!r} is not downloadable from Zenodo; generate it locally under {npz_path.parent}."
                raise RuntimeError(msg)
            url = f"https://zenodo.org/records/11471504/files/{dataset}.zip"
            download_and_extract_archive(url, str(folder), filename=f"{dataset}.zip", md5=self.md5s[dataset])

        with np.load(npz_path) as npz:
            images = npz["test_images"]  # (5N, H, W) or (5N, H, W, C) uint8
            labels = npz["test_labels"]  # (5N, 1)
        n = len(images) // 5  # images per severity
        sl = slice((severity - 1) * n, severity * n)
        self.data = np.ascontiguousarray(images[sl])  # materialize only this severity
        self.targets = labels[sl].squeeze(axis=-1).tolist()

    def __len__(self) -> int:
        """Return the number of images in this dataset/corruption/severity slice.

        Returns:
            The number of images in this dataset/corruption/severity slice.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Get the (image, label) pair at the given index.

        Args:
            index: Index within the dataset.

        Returns:
            The ``(image, label)`` tuple, with ``transform``/``target_transform`` applied. Grayscale
            flags yield mode-``L`` images and RGB flags mode-``RGB`` images.
        """
        img = Image.fromarray(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
