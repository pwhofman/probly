"""Tests for the data module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from torchvision.datasets import CIFAR10, ImageNet

import json
from pathlib import Path
import shutil
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
    MedMNISTC,
    MiceBone,
    Pig,
    Plankton,
    Synthetic,
    TinyImageNet,
    Treeversity1,
    Treeversity6,
    Turkey,
)


def patch_cifar10_init(self: CIFAR10, root: str, train: bool, transform: Callable[..., Any], download: bool) -> None:  # noqa: ARG001, the init requires these arguments
    self.root = root


def _write_fake_cifar10h_counts(root: Path, counts: np.ndarray) -> None:
    """Write fake CIFAR-10H human-annotation counts where CIFAR10H expects them."""
    path = root / CIFAR10H.cifar10h_folder / CIFAR10H.cifar10h_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), counts)


@patch("probly.datasets.torch.torchvision.datasets.CIFAR10.__init__", new=patch_cifar10_init)
def test_cifar10h(tmp_path: Path) -> None:
    _write_fake_cifar10h_counts(tmp_path, np.ones((5, 10)))
    dataset = CIFAR10H(root=str(tmp_path))
    assert torch.allclose(torch.sum(dataset.targets, dim=1), torch.ones(5))


@patch("probly.datasets.torch.torchvision.datasets.CIFAR10.__init__", new=patch_cifar10_init)
def test_cifar10h_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="download=True"):
        CIFAR10H(root=str(tmp_path))


@patch("probly.datasets.torch.download_url")
@patch("probly.datasets.torch.torchvision.datasets.CIFAR10.__init__", new=patch_cifar10_init)
def test_cifar10h_download_invoked(mock_download: MagicMock, tmp_path: Path) -> None:
    data_dir = tmp_path / CIFAR10H.cifar10h_folder
    # Simulate the download: when invoked, materialize the counts so loading succeeds.
    mock_download.side_effect = lambda *_args, **_kwargs: _write_fake_cifar10h_counts(tmp_path, np.ones((5, 10)))
    dataset = CIFAR10H(root=str(tmp_path), download=True)
    mock_download.assert_called_once_with(
        CIFAR10H.cifar10h_url, str(data_dir), filename=CIFAR10H.cifar10h_filename, md5=CIFAR10H.cifar10h_md5
    )
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


def _write_fake_tinyimagenet(folder: Path, *, grayscale_val: bool = False) -> None:
    """Write a tiny fake tiny-imagenet-200 tree with two classes, listed unsorted in wnids.txt.

    Args:
        folder: The ``tiny-imagenet-200`` directory to create.
        grayscale_val: Whether to write the first validation image as a grayscale JPEG, mirroring
            the mode-``L`` sources present in the real dataset.
    """
    wnids = ("n02124075", "n01443537")  # deliberately unsorted, as in the real wnids.txt
    (folder / "val" / "images").mkdir(parents=True, exist_ok=True)
    (folder / "wnids.txt").write_text("\n".join(wnids) + "\n")
    annotations = []
    for i, wnid in enumerate(wnids):
        images = folder / "train" / wnid / "images"
        images.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(images / f"{wnid}_0.JPEG")
        name = f"val_{i}.JPEG"
        Image.new("L" if grayscale_val and i == 0 else "RGB", (64, 64)).save(folder / "val" / "images" / name)
        annotations.append(f"{name}\t{wnid}\t0\t0\t63\t63")
    (folder / "val" / "val_annotations.txt").write_text("\n".join(annotations) + "\n")


def test_tinyimagenet_invalid_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        TinyImageNet(tmp_path, "test")  # unlabeled in the release, so intentionally unsupported


def test_tinyimagenet_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="download=True"):
        TinyImageNet(tmp_path, "train", download=False)


def test_tinyimagenet_class_order_is_sorted(tmp_path: Path) -> None:
    _write_fake_tinyimagenet(tmp_path / "tiny-imagenet-200")
    dataset = TinyImageNet(tmp_path, "train")
    # wnids.txt lists n02124075 first, but class indices follow sorted order, as ImageFolder assigns.
    assert dataset.classes == ["n01443537", "n02124075"]
    assert dataset.class_to_idx == {"n01443537": 0, "n02124075": 1}
    assert [path.parent.parent.name for path, _ in dataset.samples] == dataset.classes
    assert dataset.targets == [0, 1]


def test_tinyimagenet_partial_extraction_raises(tmp_path: Path) -> None:
    folder = tmp_path / "tiny-imagenet-200"
    _write_fake_tinyimagenet(folder)
    # An interrupted extraction can leave wnids.txt behind while class directories are missing;
    # globbing would otherwise yield a silently smaller training set.
    shutil.rmtree(folder / "train" / "n01443537")
    with pytest.raises(RuntimeError, match="Incomplete dataset: 1 of 2 train class directories"):
        TinyImageNet(tmp_path, "train")


def test_tinyimagenet_missing_val_images_raises(tmp_path: Path) -> None:
    folder = tmp_path / "tiny-imagenet-200"
    _write_fake_tinyimagenet(folder)
    shutil.rmtree(folder / "val" / "images")
    with pytest.raises(RuntimeError, match="Incomplete dataset"):
        TinyImageNet(tmp_path, "val")


def test_tinyimagenet_train_getitem(tmp_path: Path) -> None:
    _write_fake_tinyimagenet(tmp_path / "tiny-imagenet-200")
    dataset = TinyImageNet(tmp_path, "train")
    assert len(dataset) == 2
    img, target = dataset[0]
    assert img.size == (64, 64)
    assert target == 0


def test_tinyimagenet_val_targets_follow_annotations(tmp_path: Path) -> None:
    _write_fake_tinyimagenet(tmp_path / "tiny-imagenet-200")
    dataset = TinyImageNet(tmp_path, "val")
    assert len(dataset) == 2
    # val_0 is annotated n02124075 (sorted index 1) and val_1 is n01443537 (sorted index 0).
    assert dataset.targets == [1, 0]


def test_tinyimagenet_grayscale_converted_to_rgb(tmp_path: Path) -> None:
    _write_fake_tinyimagenet(tmp_path / "tiny-imagenet-200", grayscale_val=True)
    dataset = TinyImageNet(tmp_path, "val")
    img, _ = dataset[0]
    assert img.mode == "RGB"  # a mode-L source is widened so it collates with the RGB images


def test_tinyimagenet_transform_applied(tmp_path: Path) -> None:
    _write_fake_tinyimagenet(tmp_path / "tiny-imagenet-200")
    dataset = TinyImageNet(
        tmp_path,
        "train",
        transform=lambda _im: "transformed",
        target_transform=lambda target: target + 100,
    )
    img, target = dataset[0]
    assert img == "transformed"
    assert target == 100  # target_transform applied to the integer label 0


@patch("probly.datasets.torch.download_and_extract_archive")
def test_tinyimagenet_download_invoked(mock_download: MagicMock, tmp_path: Path) -> None:
    folder = tmp_path / "tiny-imagenet-200"
    # Simulate the download: when invoked, materialize the fixtures so loading succeeds.
    mock_download.side_effect = lambda *_args, **_kwargs: _write_fake_tinyimagenet(folder)
    dataset = TinyImageNet(tmp_path, "train", download=True)
    mock_download.assert_called_once_with(
        TinyImageNet.url, str(tmp_path), filename=TinyImageNet.filename, md5=TinyImageNet.zip_md5
    )
    assert len(dataset) == 2


def _write_fake_medmnistc(folder: Path, corruption: str, n_per_severity: int = 2, *, channels: int | None = 3) -> None:
    """Write tiny fake MedMNIST-C fixtures: image row i encodes its global index in pixel [0, 0].

    Args:
        folder: Directory to write ``{corruption}.npz`` into; created if missing.
        corruption: Corruption name, used as the npz filename.
        n_per_severity: Images per severity; the npz stacks ``5 * n_per_severity`` rows.
        channels: Channel count for an ``(5N, H, W, C)`` stack, or ``None`` for a grayscale
            ``(5N, H, W)`` stack (as shipped for the organ and breast flags).
    """
    folder.mkdir(parents=True, exist_ok=True)
    total = 5 * n_per_severity
    shape = (total, 8, 8) if channels is None else (total, 8, 8, channels)
    images = np.zeros(shape, dtype=np.uint8)
    for i in range(total):
        images[i, 0, 0] = i  # encodes the global row index (total < 256); broadcasts over channels
    labels = np.tile(np.arange(n_per_severity, dtype=np.int64), 5).reshape(-1, 1)  # (5N, 1)
    np.savez(str(folder / f"{corruption}.npz"), test_images=images, test_labels=labels)


def test_medmnistc_invalid_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown MedMNIST dataset"):
        MedMNISTC(tmp_path, "not_a_dataset", "pixelate", severity=1)


def test_medmnistc_invalid_corruption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown corruption"):
        MedMNISTC(tmp_path, "dermamnist", "not_a_corruption", severity=1)


def test_medmnistc_invalid_severity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="severity"):
        MedMNISTC(tmp_path, "dermamnist", "pixelate", severity=6)


def test_medmnistc_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="download=True"):
        MedMNISTC(tmp_path, "dermamnist", "pixelate", severity=1, download=False)


def test_medmnistc_slicing_and_getitem(tmp_path: Path) -> None:
    _write_fake_medmnistc(tmp_path / "medmnist_c" / "dermamnist", "pixelate", n_per_severity=2)
    dataset = MedMNISTC(tmp_path, "dermamnist", "pixelate", severity=2)  # n=2 -> global rows [2:4]
    assert len(dataset) == 2
    img, target = dataset[0]
    assert np.asarray(img)[0, 0, 0] == 2  # severity-2 slice starts at global row 2
    assert target == 0
    img1, target1 = dataset[1]
    assert np.asarray(img1)[0, 0, 0] == 3
    assert target1 == 1


def test_medmnistc_grayscale_getitem(tmp_path: Path) -> None:
    _write_fake_medmnistc(tmp_path / "medmnist_c" / "breastmnist", "pixelate", channels=None)
    dataset = MedMNISTC(tmp_path, "breastmnist", "pixelate", severity=1)
    img, _ = dataset[0]
    assert img.mode == "L"  # a grayscale (5N, H, W) stack loads as mode-L images
    assert np.asarray(img)[0, 0] == 0  # severity-1 slice starts at global row 0


def test_medmnistc_transform_applied(tmp_path: Path) -> None:
    _write_fake_medmnistc(tmp_path / "medmnist_c" / "dermamnist", "gaussian_noise")
    dataset = MedMNISTC(
        tmp_path,
        "dermamnist",
        "gaussian_noise",
        severity=1,
        transform=lambda _im: "transformed",
        target_transform=lambda target: target + 100,
    )
    img, target = dataset[0]
    assert img == "transformed"
    assert target == 100  # target_transform applied to the integer label 0


@patch("probly.datasets.torch.download_and_extract_archive")
def test_medmnistc_download_invoked(mock_download: MagicMock, tmp_path: Path) -> None:
    flag, corruption = "breastmnist", "pixelate"
    folder = tmp_path / "medmnist_c" / flag
    # Simulate the download: when invoked, materialize the fixtures so loading succeeds.
    mock_download.side_effect = lambda *_args, **_kwargs: _write_fake_medmnistc(folder, corruption, channels=None)
    dataset = MedMNISTC(tmp_path, flag, corruption, severity=1, download=True)
    expected_url = f"https://zenodo.org/records/11471504/files/{flag}.zip"
    mock_download.assert_called_once_with(
        expected_url, str(tmp_path / "medmnist_c"), filename=f"{flag}.zip", md5=MedMNISTC.md5s[flag]
    )
    assert len(dataset) == 2


def test_medmnistc_covers_all_datasets() -> None:
    # Eleven single-label MedMNIST2D datasets are supported; ChestMNIST is intentionally excluded
    # because it is multi-label while this loader returns hard integer labels.
    assert len(MedMNISTC.corruptions) == 11
    assert "chestmnist" not in MedMNISTC.corruptions
    # Every supported flag is downloadable except tissuemnist, which is registry-only (not on Zenodo).
    assert set(MedMNISTC.md5s) == set(MedMNISTC.corruptions) - {"tissuemnist"}
    assert "tissuemnist" in MedMNISTC.corruptions


@patch("probly.datasets.torch.download_and_extract_archive")
def test_medmnistc_registry_only_download_raises(mock_download: MagicMock, tmp_path: Path) -> None:
    # tissuemnist is a valid flag but is not on Zenodo, so download must fail clearly (not KeyError).
    with pytest.raises(RuntimeError, match="not downloadable"):
        MedMNISTC(tmp_path, "tissuemnist", "pixelate", severity=1, download=True)
    mock_download.assert_not_called()
