import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image


class CIFAR10H(torchvision.datasets.CIFAR10):
    def __init__(self, root, transform=None, download=False):
        super().__init__(root, train=False, transform=transform, download=download)
        first_order_path = os.path.join(self.root, 'cifar-10h-master/data/cifar10h-counts.npy')
        self.counts = np.load(first_order_path)
        self.counts = torch.tensor(self.counts, dtype=torch.float32)
        self.targets = self.counts / self.counts.sum(dim=1, keepdim=True)

class DCICDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, first_order=True):

        root = os.path.expanduser(root)
        with open(os.path.join(root, 'annotations.json')) as f:
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
        self.label_mappings = {label: idx for idx, label in enumerate(set(label for labels in self.image_labels.values() for label in labels))}
        self.num_classes = len(set(label for labels in self.image_labels.values() for label in labels))

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform:
            image = self.transform(image)
        target = self.targets[index]
        return image, target

class Benthic(DCICDataset):
    def __init__(self, root, transform=None, first_order=True):
        super().__init__(os.path.join(root, 'Benthic'), transform, first_order)
