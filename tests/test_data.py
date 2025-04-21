"""Tests for the data module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import torch

from probly.data import CIFAR10H, Benthic, ImageNetReaL, Plankton, Treeversity1, Treeversity6


def patch_cifar10_init(self, root, train, transform, download):  # noqa: ARG001, the init requires these arguments
    self.root = root


@patch("probly.data.np.load")
@patch("probly.data.torchvision.datasets.CIFAR10.__init__", new=patch_cifar10_init)
def test_cifar10h(mock_np_load, tmp_path):
    counts = np.ones((5, 10))
    mock_np_load.return_value = counts
    dataset = CIFAR10H(root=str(tmp_path))
    dataset.data = [np.zeros((3, 32, 32))] * 2
    assert torch.allclose(torch.sum(dataset.targets, dim=1), torch.ones(5))


def patch_imagenet_init(self, root, split, transform):  # noqa: ARG001, the init requires these arguments
    self.samples = [
        ("some/path/ILSVRC2012_val_00000001.JPEG", 0),
        ("some/path/ILSVRC2012_val_00000002.JPEG", 1),
        ("some/path/ILSVRC2012_val_00000003.JPEG", 2),
    ]
    self.classes = [0, 1, 2]


@patch("probly.data.torchvision.datasets.ImageNet.__init__", new=patch_imagenet_init)
@patch("pathlib.Path.open", new_callable=mock_open, read_data=json.dumps([[], [1], [1, 2]]))
def test_imagenetreal(tmp_path):
    dataset = ImageNetReaL(str(tmp_path))
    for dist in dataset.dists:
        assert torch.isclose(torch.sum(dist), torch.tensor(1.0))


@patch("probly.data.DCICDataset.__init__", return_value=None)
def test_benthic(mock_dcic_init):
    root = "some/path"
    _ = Benthic(root, first_order=False)
    expected = Path(root) / "Benthic"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.data.DCICDataset.__init__", return_value=None)
def test_plankton(mock_dcic_init):
    root = "some/path"
    _ = Plankton(root, first_order=False)
    expected = Path(root) / "Plankton"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.data.DCICDataset.__init__", return_value=None)
def test_treeversity1(mock_dcic_init):
    root = "some/path"
    _ = Treeversity1(root, first_order=False)
    expected = Path(root) / "Treeversity#1"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.data.DCICDataset.__init__", return_value=None)
def test_treeversity6(mock_dcic_init):
    root = "some/path"
    _ = Treeversity6(root, first_order=False)
    expected = Path(root) / "Treeversity#6"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)
