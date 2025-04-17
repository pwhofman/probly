"""Collection of dataset classes for loading data from different datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch
import torchvision

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class CIFAR10H(torchvision.datasets.CIFAR10):
    """A Dataset class for the CIFAR10H dataset introduced in https://arxiv.org/abs/1908.07086.

    The dataset can be found at https://github.com/jcpeterson/cifar-10h.

    Attributes:
        counts: torch.Tensor,
        targets: torch.Tensor size (n_instances, n_classes), first-order distribution
    """

    def __init__(self, root: str, transform: Callable | None = None, *, download: bool = False) -> None:
        """Initialize an instance of the CIFAR10H class.

        Args:
            root: str, root directory of the dataset
            transform: optional transform to apply to the data
            download: bool, whether to download the CIFAR10 dataset or not
        """
        super().__init__(root, train=False, transform=transform, download=download)
        first_order_path = Path(self.root) / "cifar-10h-master" / "data" / "cifar10h-counts.npy"
        self.counts = np.load(first_order_path)
        self.counts = torch.tensor(self.counts, dtype=torch.float32)
        self.targets = self.counts / self.counts.sum(dim=1, keepdim=True)


class ImageNetReaL(torchvision.datasets.ImageNet):
    """A Dataset class for the ImageNet ReaL dataset introduced in https://arxiv.org/abs/2006.07159.

    This dataset is a re-labeled version of the ImageNet validation set, where each image can belong
    to multiple classes resulting in a distribution over classes.
    The ImageNet dataset needs to be downloaded from https://www.image-net.org and the first order labels can be
    downloaded from https://github.com/google-research/reassessed-imagenet.

    Attributes:
        dists: list, list of distributions over target classes.
    """

    def __init__(self, root: str, transform: Callable | None = None) -> None:
        """Initialize an instance of the ImageNetReaL class.

        Args:
            root: str, root directory of the dataset
            transform: optional transform to apply to the data
        """
        super().__init__(root=root, split="val", transform=transform)
        root = Path(root).expanduser()
        with (Path(root).expanduser() / "reassessed-imagenet-master/real.json").open() as f:
            real = json.load(f)
        real_labels = {f"ILSVRC2012_val_{(i + 1):08d}.JPEG": labels for i, labels in enumerate(real)}
        self.dists = []
        for img, _ in self.samples:
            labels = real_labels[img.split("/")[-1]]
            dist = torch.zeros(len(self.classes))
            dist[labels] = 1
            dist = dist / dist.sum()
            self.dists.append(dist)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the item at the specified index.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, dist) where dist is a distribution over target classes.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        dist = self.dists[index]
        return sample, dist


class DCICDataset(torch.utils.data.Dataset):
    """A Dataset base class for the DCICDatasets introduced in https://arxiv.org/abs/2207.06214.

    These datasets can be found at https://zenodo.org/records/7180818.

    Attributes:
        root: str, root directory of the dataset
        transform: transform to apply to the data
        image_labels: dict, dictionary of image labels grouped by image
        image_paths: list, image paths
        label_mappings: dict, # TODO
        num_classes: int, number of classes
        data: list, images
        targets: list, labels
        # TODO remove unnecessary fields
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the DCICDataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
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
        self.label_mappings = {
            label: idx
            for idx, label in enumerate(
                {label for labels in self.image_labels.values() for label in labels}
                # TODO simplify code here
            )
        }
        self.num_classes = len({label for labels in self.image_labels.values() for label in labels})

        self.data = []
        self.targets = []
        for img_path in self.image_paths:
            full_img_path = Path(self.root) / img_path
            image = Image.open(full_img_path).convert("RGB").copy()  # TODO: optimize this loading process
            self.data.append(image)
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
            int, The number of instances in the dataset.

        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returned indexed item in the dataset.

        Args:
            index: int, Index within the dataset.

        Returns:
            (image, target): tuple[torch.Tensor, torch.Tensor], The image and label within the dataset.

        """
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        target = self.targets[index]
        return image, target


class Benthic(DCICDataset):
    """Implementation of the Benthic dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the Benthic dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "Benthic", transform, first_order=first_order)


class Plankton(DCICDataset):
    """Implementation of the Plankton dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the Plankton dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "Plankton", transform, first_order=first_order)


class QualityMRI(DCICDataset):
    """Implementation of the QualityMRI dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the QualityMRI dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "QualityMRI", transform, first_order=first_order)


class Treeversity1(DCICDataset):
    """Implementation of the Treeversity#1 dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the Treeversity#1 dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "Treeversity#1", transform, first_order=first_order)


class Treeversity6(DCICDataset):
    """Implementation of the Treeversity#6 dataset.

    The dataset can be found at https://zenodo.org/records/7180818.
    """

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the Treeversity#6 dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "Treeversity#6", transform, first_order=first_order)
