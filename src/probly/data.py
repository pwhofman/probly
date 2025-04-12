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


class CIFAR10H(torchvision.datasets.CIFAR10):
    """A Dataset class for the CIFAR10H dataset.

    The dataset can be found at https://zenodo.org/records/7180818.

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


class DCICDataset(torch.utils.data.Dataset):
    """A Dataset class for the DCICDataset.

    The dataset can be found at https://zenodo.org/records/7180818.

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
            image = Image.open(full_img_path)
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
    """Implementation of the Benthic dataset."""

    def __init__(self, root: Path | str, transform: Callable | None = None, *, first_order: bool = True) -> None:
        """Initialize an instance of the Benthic dataset class.

        Args:
            root: Path or str, root directory of the dataset
            transform: optional transform to apply to the data
            first_order: bool, whether to use first order data or class labels
        """
        super().__init__(Path(root) / "Benthic", transform, first_order=first_order)
