import json
import os
from collections.abc import Callable

import numpy as np
import torch
import torchvision
from PIL import Image


class CIFAR10H(torchvision.datasets.CIFAR10):
    def __init__(
        self, root: str, transform: Callable | None = None, download: bool = False
    ) -> None:
        super().__init__(root, train=False, transform=transform, download=download)
        first_order_path = os.path.join(self.root, "cifar-10h-master/data/cifar10h-counts.npy")
        self.counts = np.load(first_order_path)
        self.counts = torch.tensor(self.counts, dtype=torch.float32)
        self.targets = self.counts / self.counts.sum(dim=1, keepdim=True)


class DCICDataset(torch.utils.data.Dataset):
    """A Dataset class for the DCICDataset.

    Args:
        root: str, root directory of the dataset
        transform: optional transform to apply to the data
        first_order: bool, whether to use first order data or not

    """

    def __init__(
        self, root: str, transform: Callable | None = None, first_order: bool = True
    ) -> None:
        root = os.path.expanduser(root)
        with open(os.path.join(root, "annotations.json")) as f:
            annotations = json.load(f)

        self.root = os.path.dirname(root)
        self.transform = transform
        self.image_labels = {}  # Dictionary to group annotations by image

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
                set(label for labels in self.image_labels.values() for label in labels)
            )
        }
        self.num_classes = len(
            set(label for labels in self.image_labels.values() for label in labels)
        )

        self.data = []
        self.targets = []
        for img_path in self.image_paths:
            full_img_path = os.path.join(self.root, img_path)
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
    def __init__(
        self, root: str, transform: Callable | None = None, first_order: bool = True
    ) -> None:
        super().__init__(os.path.join(root, "Benthic"), transform, first_order)
