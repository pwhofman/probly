"""Tests for the data module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torchvision.datasets import CIFAR10, ImageNet

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
from PIL import Image
import pytest
import torch

from probly.datasets.torch import (
    CIFAR10C,
    CIFAR10H,
    CIFAR10HDCIC,
    Benthic,
    DCICDataset,
    ImageNetReaL,
    MiceBone,
    Pig,
    Plankton,
    Synthetic,
    Treeversity1,
    Treeversity6,
    Turkey,
)


def patch_cifar10_init(self: CIFAR10, root: str, train: bool, transform: Callable[..., Any], download: bool) -> None:  # noqa: ARG001, the init requires these arguments
    self.root = root


@patch("probly.datasets.torch.np.load")
@patch("probly.datasets.torch.torchvision.datasets.CIFAR10.__init__", new=patch_cifar10_init)
def test_cifar10h(mock_np_load: MagicMock, tmp_path: Path) -> None:
    counts = np.ones((5, 10))
    mock_np_load.return_value = counts
    dataset = CIFAR10H(root=str(tmp_path))
    dataset.data = [np.zeros((3, 32, 32))] * 2
    assert torch.allclose(torch.sum(dataset.targets, dim=1), torch.ones(5))


def patch_imagenet_init(model: ImageNet, root: str, split: str, transform: Callable[..., Any]) -> None:  # noqa: ARG001, the init requires these arguments
    model.samples = [
        ("some/path/ILSVRC2012_val_00000001.JPEG", 0),
        ("some/path/ILSVRC2012_val_00000002.JPEG", 1),
        ("some/path/ILSVRC2012_val_00000003.JPEG", 2),
    ]
    model.classes = [0, 1, 2]  # ty:ignore[invalid-assignment]


@patch("probly.datasets.torch.torchvision.datasets.ImageNet.__init__", new=patch_imagenet_init)
@patch("pathlib.Path.open", new_callable=mock_open, read_data=json.dumps([[], [1], [1, 2]]))
def test_imagenetreal(tmp_path: Path) -> None:
    dataset = ImageNetReaL(str(tmp_path))
    for dist in dataset.dists:
        assert torch.isclose(torch.sum(dist), torch.tensor(1.0))
    # __getitem__ lazily loads via self.loader and applies the transform.
    dataset.loader = lambda _path: Image.new("RGB", (2, 2))
    dataset.transform = lambda sample: sample
    sample, dist = dataset[0]
    assert isinstance(sample, Image.Image)
    assert torch.isclose(torch.sum(dist), torch.tensor(1.0))


def _write_fake_dcic(root: Path) -> None:
    """Write a tiny DCIC dataset (annotations.json + two images) under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    # image_path is resolved relative to root.parent (DCICDataset sets self.root = root.parent).
    annotations = [
        {
            "annotations": [
                {"image_path": f"{root.name}/img0.png", "class_label": 0},
                {"image_path": f"{root.name}/img0.png", "class_label": 1},
                {"image_path": f"{root.name}/img1.png", "class_label": 1},
            ]
        }
    ]
    with (root / "annotations.json").open("w") as f:
        json.dump(annotations, f)
    for name in ("img0.png", "img1.png"):
        Image.new("RGB", (4, 4)).save(root / name)


def test_dcic_dataset_first_order(tmp_path: Path) -> None:
    _write_fake_dcic(tmp_path / "DS")
    dataset = DCICDataset(tmp_path / "DS", first_order=True)
    assert len(dataset) == 2
    image, target = dataset[0]
    assert isinstance(image, Image.Image)
    assert image.size == (4, 4)
    assert target.shape == (dataset.num_classes,)
    assert torch.isclose(target.sum(), torch.tensor(1.0))


def test_dcic_dataset_class_labels_with_transform(tmp_path: Path) -> None:
    _write_fake_dcic(tmp_path / "DS")
    dataset = DCICDataset(tmp_path / "DS", transform=lambda _img: torch.zeros(3), first_order=False)
    image, target = dataset[1]
    assert torch.equal(image, torch.zeros(3))  # transform applied
    assert target.ndim == 0  # a single sampled class index


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_cifar10h_dcic(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = CIFAR10HDCIC(root, first_order=False)
    expected = Path(root) / "CIFAR10H"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_benthic(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Benthic(root, first_order=False)
    expected = Path(root) / "Benthic"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_plankton(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Plankton(root, first_order=False)
    expected = Path(root) / "Plankton"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_micebone(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = MiceBone(root, first_order=False)
    expected = Path(root) / "MiceBone"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_pig(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Pig(root, first_order=False)
    expected = Path(root) / "Pig"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_synthetic(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Synthetic(root, first_order=False)
    expected = Path(root) / "Synthetic"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_turkey(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Turkey(root, first_order=False)
    expected = Path(root) / "Turkey"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_treeversity1(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Treeversity1(root, first_order=False)
    expected = Path(root) / "Treeversity#1"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


@patch("probly.datasets.torch.DCICDataset.__init__", return_value=None)
def test_treeversity6(mock_dcic_init: MagicMock) -> None:
    root = "some/path"
    _ = Treeversity6(root, first_order=False)
    expected = Path(root) / "Treeversity#6"
    mock_dcic_init.assert_called_once_with(expected, None, first_order=False)


def _write_fake_cifar10c(folder: Path, corruption: str, n_per_severity: int = 2) -> None:
    """Write tiny fake CIFAR-10-C fixtures: data row i encodes its global index in pixel [0,0,0]."""
    folder.mkdir(parents=True, exist_ok=True)
    total = 5 * n_per_severity
    data = np.zeros((total, 32, 32, 3), dtype=np.uint8)
    for i in range(total):
        data[i, 0, 0, 0] = i  # encodes the global row index (total < 256)
    labels = np.tile(np.arange(n_per_severity, dtype=np.int64), 5)  # identical across severities
    np.save(folder / f"{corruption}.npy", data)
    np.save(folder / "labels.npy", labels)


def test_cifar10c_invalid_corruption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        CIFAR10C(tmp_path, "not_a_corruption", severity=1)


def test_cifar10c_invalid_severity(tmp_path: Path) -> None:
    _write_fake_cifar10c(tmp_path / "CIFAR-10-C", "gaussian_noise")
    with pytest.raises(ValueError, match="severity"):
        CIFAR10C(tmp_path, "gaussian_noise", severity=6)


def test_cifar10c_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="download=True"):
        CIFAR10C(tmp_path, "gaussian_noise", severity=1, download=False)


def test_cifar10c_slicing_and_getitem(tmp_path: Path) -> None:
    _write_fake_cifar10c(tmp_path / "CIFAR-10-C", "gaussian_noise", n_per_severity=2)
    dataset = CIFAR10C(tmp_path, "gaussian_noise", severity=2)  # n=2 -> global rows [2:4]
    assert len(dataset) == 2
    img, target = dataset[0]
    assert np.asarray(img)[0, 0, 0] == 2  # severity-2 slice starts at global row 2
    assert target == 0
    img1, target1 = dataset[1]
    assert np.asarray(img1)[0, 0, 0] == 3
    assert target1 == 1


def test_cifar10c_transform_applied(tmp_path: Path) -> None:
    _write_fake_cifar10c(tmp_path / "CIFAR-10-C", "fog")
    dataset = CIFAR10C(
        tmp_path,
        "fog",
        severity=1,
        transform=lambda _im: "transformed",
        target_transform=lambda target: target + 100,
    )
    img, target = dataset[0]
    assert img == "transformed"
    assert target == 100  # target_transform applied to the integer label 0


@patch("probly.datasets.torch.download_and_extract_archive")
def test_cifar10c_download_invoked(mock_download: MagicMock, tmp_path: Path) -> None:
    folder = tmp_path / "CIFAR-10-C"
    # Simulate the download: when invoked, materialize the fixtures so loading succeeds.
    mock_download.side_effect = lambda *_args, **_kwargs: _write_fake_cifar10c(folder, "snow")
    dataset = CIFAR10C(tmp_path, "snow", severity=1, download=True)
    mock_download.assert_called_once_with(CIFAR10C.url, str(tmp_path), filename=CIFAR10C.filename, md5=CIFAR10C.tar_md5)
    assert len(dataset) == 2
