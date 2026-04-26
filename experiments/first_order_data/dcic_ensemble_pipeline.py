from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

from entmax import entmax_bisect
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import get_model, get_model_weights

from probly.datasets.torch import (
    CIFAR10HDCIC,
    Benthic,
    DCICDataset,
    MiceBone,
    Pig,
    Plankton,
    QualityMRI,
    Synthetic,
    Treeversity1,
    Treeversity6,
    Turkey,
)
from probly.method.ensemble import ensemble
from probly.quantification import quantify
from probly.representation.distribution import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)
from probly.utils.torch import torch_collect_outputs

torch.set_float32_matmul_precision("high")

DEFAULT_ENCODERS = {
    "resnet18": 224,
    "resnet50": 224,
    "wide_resnet50_2": 224,
    "resnext50_32x4d": 224,
    "densenet121": 224,
    "efficientnet_b0": 224,
    "convnext_tiny": 224,
    "vit_b_16": 224,
}

DCIC_DATASET_LOADERS = {
    "Benthic": Benthic,
    "CIFAR10H": CIFAR10HDCIC,
    "MiceBone": MiceBone,
    "Pig": Pig,
    "Plankton": Plankton,
    "QualityMRI": QualityMRI,
    "Synthetic": Synthetic,
    "Turkey": Turkey,
    "Treeversity#1": Treeversity1,
    "Treeversity#6": Treeversity6,
}


@dataclass(slots=True)
class FoldResult:
    test_fold: str
    train_size: int
    val_size: int
    test_size: int
    member_cross_entropies: list[float]
    ensemble_cross_entropy: float
    mean_uncertainty: dict[str, float]
    prediction_file: str
    member_prediction_files: list[str]
    model_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_fold": self.test_fold,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "member_cross_entropies": self.member_cross_entropies,
            "ensemble_cross_entropy": self.ensemble_cross_entropy,
            "mean_uncertainty": self.mean_uncertainty,
            "prediction_file": self.prediction_file,
            "member_prediction_files": self.member_prediction_files,
            "model_files": self.model_files,
        }


@dataclass(slots=True)
class DatasetResult:
    dataset_name: str
    encoder_name: str
    class_names: list[str]
    folds: list[FoldResult]
    mean_member_cross_entropy: float
    mean_ensemble_cross_entropy: float
    mean_uncertainty: dict[str, float]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "encoder_name": self.encoder_name,
            "class_names": self.class_names,
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_member_cross_entropy": self.mean_member_cross_entropy,
            "mean_ensemble_cross_entropy": self.mean_ensemble_cross_entropy,
            "mean_uncertainty": self.mean_uncertainty,
            "config": self.config,
        }


@dataclass(frozen=True, slots=True)
class EntmaxImageExperimentConfig:
    data_root: Path
    output_root: Path
    encoder_name: str
    pretrained: bool
    freeze_encoder: bool
    ensemble_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_size: float
    early_stopping_patience: int
    num_workers: int
    device: str
    seed: int
    test_fold: str
    augmentation: str
    classifier_dropout: float
    entmax_alpha: float


class SoftTargetEntmaxBisectLoss(nn.Module):
    def __init__(self, alpha: float = 1.5, n_iter: int = 50):
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = entmax_bisect(
            logits,
            alpha=self.alpha,
            dim=1,
            n_iter=self.n_iter,  # number iterations to solve for entmax normalization threshold
        )
        # The loss is derived from the Fenchel-Young loss formulation for entmax.
        # This is necessary since Entmax uses a different mapping from logits to probabilities,
        # so it has a different matched loss.
        # This is equivalent to the standard cross-entropy loss when alpha=1 (softmax)
        omega = (
            (1 - (probabilities**self.alpha).sum(dim=1)) / (self.alpha * (self.alpha - 1))
        )  # Tsallis entropy, acts as regularization to prevent solution becoming too sharp too early, rewards distributions with larger entropy
        return (omega + ((probabilities - targets) * logits).sum(dim=1)).mean()


def uses_softmax_training(entmax_alpha: float) -> bool:
    return math.isclose(entmax_alpha, 1.0)


class ImageEntmaxClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        pretrained: bool,
        freeze_encoder: bool,
        classifier_dropout: float,
    ):
        super().__init__()

        weights = None
        if pretrained:
            weights = get_model_weights(encoder_name).DEFAULT

        self.encoder = get_model(encoder_name, weights=weights)
        in_features = replace_classification_head_with_identity(
            self.encoder,
            encoder_name,
        )
        self.head = build_classifier_head(
            in_features=in_features,
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.head(features)


def build_classifier_head(
    *,
    in_features: int,
    num_classes: int,
    classifier_dropout: float,
) -> nn.Module:
    if classifier_dropout > 0:
        return nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(in_features, num_classes),
        )
    return nn.Linear(in_features, num_classes)


def replace_classification_head_with_identity(
    model: nn.Module,
    encoder_name: str,
) -> int:
    if encoder_name in {"resnet18", "resnet50", "wide_resnet50_2", "resnext50_32x4d"}:
        in_features = int(model.fc.in_features)
        model.fc = nn.Identity()
        return in_features

    if encoder_name == "densenet121":
        in_features = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return in_features

    if encoder_name in {"efficientnet_b0", "convnext_tiny"}:
        in_features = int(model.classifier[-1].in_features)
        model.classifier = nn.Identity()
        return in_features

    if encoder_name == "vit_b_16":
        in_features = int(model.heads.head.in_features)
        model.heads = nn.Identity()
        return in_features

    raise ValueError(f"Unsupported encoder: {encoder_name}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def list_image_datasets(data_root: Path) -> list[str]:
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def config_to_dict(config: EntmaxImageExperimentConfig) -> dict[str, Any]:
    return {
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "encoder_name": config.encoder_name,
        "pretrained": config.pretrained,
        "freeze_encoder": config.freeze_encoder,
        "ensemble_size": config.ensemble_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "validation_size": config.validation_size,
        "early_stopping_patience": config.early_stopping_patience,
        "num_workers": config.num_workers,
        "device": config.device,
        "seed": config.seed,
        "test_fold": config.test_fold,
        "augmentation": config.augmentation,
        "classifier_dropout": config.classifier_dropout,
        "entmax_alpha": config.entmax_alpha,
    }


def sanitize_path_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value).strip(
        "-"
    )


def build_run_name(config: EntmaxImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "linear"
    pretrained = "pretrained" if config.pretrained else "scratch"
    parts = [
        config.encoder_name,
        "softmax" if uses_softmax_training(config.entmax_alpha) else "entmax",
        mode,
        pretrained,
        f"ens{config.ensemble_size}",
        f"ep{config.epochs}",
        f"bs{config.batch_size}",
        f"lr{config.learning_rate:g}",
        f"wd{config.weight_decay:g}",
        f"seed{config.seed}",
        f"test-{config.test_fold}",
        f"alpha-{config.entmax_alpha:g}",
    ]
    if config.augmentation != "basic":
        parts.append(f"aug-{config.augmentation}")
    if config.classifier_dropout > 0:
        parts.append(f"drop-{config.classifier_dropout:g}")
    return "_".join(sanitize_path_token(part) for part in parts)


def extract_fold_name(image_path: str) -> str:
    parts = Path(image_path).parts
    return parts[1] if len(parts) > 1 else "unknown_fold"


def class_names_from_dataset(dataset: DCICDataset) -> list[str]:
    return [str(label) for label, _ in sorted(dataset.label_mappings.items(), key=lambda item: item[1])]


def load_dcic_dataset(dataset_dir: Path) -> tuple[type[DCICDataset], list[str]]:
    dataset_loader = DCIC_DATASET_LOADERS.get(dataset_dir.name)
    if dataset_loader is None:
        raise ValueError(f"Dataset '{dataset_dir.name}' is not supported by probly.datasets.")
    dataset = dataset_loader(dataset_dir.parent)
    return dataset_loader, class_names_from_dataset(dataset)


def build_dataset(
    dataset_name: str,
    data_root: Path,
    transform: transforms.Compose | None = None,
) -> DCICDataset:
    dataset_loader = DCIC_DATASET_LOADERS.get(dataset_name)
    if dataset_loader is None:
        raise ValueError(f"Dataset '{dataset_name}' is not supported by probly.datasets.")
    return dataset_loader(data_root, transform=transform)


def split_train_validation_indices(
    indices: list[int],
    validation_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(len(indices))
    split_index = int(len(indices) * (1 - validation_size))
    shuffled_indices = [indices[position] for position in shuffled_positions]
    return shuffled_indices[:split_index], shuffled_indices[split_index:]


def build_transform(
    encoder_name: str,
    *,
    train: bool,
    augmentation: str,
) -> transforms.Compose:
    input_size = DEFAULT_ENCODERS[encoder_name]
    steps: list[Any] = []

    if train and augmentation == "basic":
        steps.append(transforms.RandomHorizontalFlip())

    steps.extend(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    return transforms.Compose(steps)


def make_dataloader(
    dataset: Subset[DCICDataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def cross_entropy_per_sample(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)  # Prevent log(0)
    safe_targets = np.asarray(targets, dtype=np.float64)
    return -(safe_targets * np.log(clipped)).sum(axis=1)


def entropy_per_sample(distributions: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(distributions, dtype=np.float64), 1e-8, 1.0)  # Prevent log(0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def mean_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(cross_entropy_per_sample(targets, probabilities).mean())


def export_prediction_frame(
    output_path: Path,
    image_paths: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    fold_name: str,
    history: dict[str, list[float]] | None,
    uncertainty: dict[str, np.ndarray] | None = None,
) -> None:
    fieldnames = ["image_path", "fold", "cross_entropy", "target_entropy"]
    uncertainty_keys = sorted(uncertainty) if uncertainty else []
    fieldnames.extend(uncertainty_keys)
    fieldnames.extend(f"target::{class_name}" for class_name in class_names)
    fieldnames.extend(f"pred::{class_name}" for class_name in class_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_cross_entropy = cross_entropy_per_sample(targets, probabilities)
    sample_entropy = entropy_per_sample(targets)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, image_path in enumerate(image_paths):
            row: dict[str, Any] = {
                "image_path": image_path,
                "fold": fold_name,
                "cross_entropy": float(sample_cross_entropy[index]),
                "target_entropy": float(sample_entropy[index]),
            }
            for key in uncertainty_keys:
                row[key] = float(uncertainty[key][index])
            for class_index, class_name in enumerate(class_names):
                row[f"target::{class_name}"] = float(targets[index, class_index])
                row[f"pred::{class_name}"] = float(probabilities[index, class_index])
            writer.writerow(row)

    if history is not None:
        write_json(output_path.with_name("history.json"), history)


def save_results_summary(output_root: Path, results: list[DatasetResult]) -> None:
    write_json(output_root / "results.json", [result.to_dict() for result in results])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_items = 0

    with torch.set_grad_enabled(is_training):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

    return total_loss / max(total_items, 1)


def train_single_model(
    *,
    model: nn.Module,
    train_dataset: Subset[DCICDataset],
    val_dataset: Subset[DCICDataset],
    config: EntmaxImageExperimentConfig,
    seed: int,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)

    train_loader = make_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),  # small optimization, can speed up data transfer to GPU
    )
    val_loader = make_dataloader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),  # small optimization, can speed up data transfer to GPU
    )

    model = model.to(config.device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion: nn.Module
    if uses_softmax_training(config.entmax_alpha):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftTargetEntmaxBisectLoss(alpha=config.entmax_alpha)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(config.epochs):
        train_loss = run_epoch(model, train_loader, criterion, config.device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, config.device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_dataset(
    *,
    model: nn.Module,
    dataset: Subset[DCICDataset],
    config: EntmaxImageExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = make_dataloader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    model.eval()
    # Uses probly's utility to collect outputs across dataset
    logits, targets = torch_collect_outputs(model, loader, config.device)
    if uses_softmax_training(config.entmax_alpha):
        probabilities = torch.softmax(logits.float(), dim=1)
    else:
        probabilities = entmax_bisect(
            logits.float(),
            alpha=config.entmax_alpha,
            dim=1,
            n_iter=50,
            ensure_sum_one=True,
        )
    image_paths = [dataset.dataset.image_paths[index] for index in dataset.indices]

    return (
        probabilities.cpu().numpy().astype(np.float32, copy=False),
        targets.cpu().numpy(),
        image_paths,
    )


def train_single_member(
    *,
    model: nn.Module,
    member_index: int,
    member_seed: int,
    member_dir: Path,
    test_fold: str,
    train_dataset: Subset[DCICDataset],
    val_dataset: Subset[DCICDataset],
    test_dataset: Subset[DCICDataset],
    class_names: list[str],
    config: EntmaxImageExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    model, history = train_single_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        seed=member_seed,
    )
    probabilities, targets, image_paths = predict_dataset(
        model=model,
        dataset=test_dataset,
        config=config,
    )
    member_loss = mean_cross_entropy(targets, probabilities)

    model_path = member_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    prediction_path = member_dir / "predictions.csv"
    export_prediction_frame(
        output_path=prediction_path,
        image_paths=image_paths,
        targets=targets,
        probabilities=probabilities,
        class_names=class_names,
        fold_name=test_fold,
        history=history,
    )

    summary = {
        "member_index": member_index,
        "member_seed": member_seed,
        "cross_entropy": member_loss,
        "model_file": str(model_path),
        "prediction_file": str(prediction_path),
        "history_file": str(prediction_path.with_name("history.json")),
    }
    write_json(member_dir / "summary.json", summary)
    return probabilities, targets, summary, image_paths


def run_dataset_experiment(
    dataset_name: str,
    config: EntmaxImageExperimentConfig,
) -> DatasetResult:
    output_dir = config.output_root / dataset_name / config.encoder_name
    fold_dir = output_dir / config.test_fold
    fold_dir.mkdir(parents=True, exist_ok=True)

    dataset_loader, class_names = load_dcic_dataset(config.data_root / dataset_name)
    split_dataset = dataset_loader(config.data_root)
    fold_indices: dict[str, list[int]] = {}
    for index, image_path in enumerate(split_dataset.image_paths):
        fold_name = extract_fold_name(image_path)
        fold_indices.setdefault(fold_name, []).append(index)

    test_indices = fold_indices.get(config.test_fold, [])

    if not test_indices:
        raise ValueError(f"Fold '{config.test_fold}' was not found in dataset '{dataset_name}'.")

    train_indices = [
        index for fold_name, indices in fold_indices.items() if fold_name != config.test_fold for index in indices
    ]
    train_indices, val_indices = split_train_validation_indices(
        train_indices,
        validation_size=config.validation_size,
        seed=config.seed,
    )
    train_dataset = Subset(
        build_dataset(
            dataset_name,
            config.data_root,
            transform=build_transform(
                config.encoder_name,
                train=True,
                augmentation=config.augmentation,
            ),
        ),
        train_indices,
    )
    val_dataset = Subset(
        build_dataset(
            dataset_name,
            config.data_root,
            transform=build_transform(
                config.encoder_name,
                train=False,
                augmentation=config.augmentation,
            ),
        ),
        val_indices,
    )
    test_dataset = Subset(
        build_dataset(
            dataset_name,
            config.data_root,
            transform=build_transform(
                config.encoder_name,
                train=False,
                augmentation=config.augmentation,
            ),
        ),
        test_indices,
    )

    # Builds the ensemble via probly
    set_seed(config.seed)
    base_model = ImageEntmaxClassifier(
        encoder_name=config.encoder_name,
        num_classes=len(class_names),
        pretrained=config.pretrained,
        freeze_encoder=config.freeze_encoder,
        classifier_dropout=config.classifier_dropout,
    )
    ensemble_members = ensemble(
        base_model,
        num_members=config.ensemble_size,
        reset_params=not config.pretrained,
    )

    member_probabilities: list[np.ndarray] = []
    member_losses: list[float] = []
    member_prediction_files: list[str] = []
    model_files: list[str] = []

    for member_index, member in enumerate(ensemble_members):
        member_seed = config.seed + member_index
        member_dir = fold_dir / f"member_{member_index:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)

        probabilities, targets, member_summary, image_paths = train_single_member(
            model=member,
            member_index=member_index,
            member_seed=member_seed,
            member_dir=member_dir,
            test_fold=config.test_fold,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            class_names=class_names,
            config=config,
        )
        member_probabilities.append(probabilities)
        member_losses.append(float(member_summary["cross_entropy"]))
        member_prediction_files.append(member_summary["prediction_file"])
        model_files.append(member_summary["model_file"])

    member_stack = np.stack(member_probabilities, axis=0)  # (n_members, n, n_classes)
    ensemble_probabilities = member_stack.mean(axis=0)
    ensemble_loss = mean_cross_entropy(targets, ensemble_probabilities)

    # treat the stack as a sample from a second-order categorical distribution and use
    # probly to decompose uncertainty into total, aleatoric and epistemic
    ensemble_sample = ArrayCategoricalDistributionSample(
        array=ArrayCategoricalDistribution(member_stack),
        sample_axis=0,
    )
    decomposition = quantify(ensemble_sample)
    uncertainty = {
        "total_uncertainty": np.asarray(decomposition.total, dtype=np.float64),
        "aleatoric_uncertainty": np.asarray(decomposition.aleatoric, dtype=np.float64),
        "epistemic_uncertainty": np.asarray(decomposition.epistemic, dtype=np.float64),
    }
    mean_uncertainty = {name: float(values.mean()) for name, values in uncertainty.items()}

    ensemble_prediction_path = fold_dir / "ensemble_predictions.csv"
    export_prediction_frame(
        output_path=ensemble_prediction_path,
        image_paths=image_paths,
        targets=targets,
        probabilities=ensemble_probabilities,
        class_names=class_names,
        fold_name=config.test_fold,
        history=None,
        uncertainty=uncertainty,
    )

    fold_result = FoldResult(
        test_fold=config.test_fold,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        member_cross_entropies=member_losses,
        ensemble_cross_entropy=ensemble_loss,
        mean_uncertainty=mean_uncertainty,
        prediction_file=str(ensemble_prediction_path),
        member_prediction_files=member_prediction_files,
        model_files=model_files,
    )
    write_json(fold_dir / "summary.json", fold_result.to_dict())

    result = DatasetResult(
        dataset_name=dataset_name,
        encoder_name=config.encoder_name,
        class_names=class_names,
        folds=[fold_result],
        mean_member_cross_entropy=float(np.mean(member_losses)),
        mean_ensemble_cross_entropy=ensemble_loss,
        mean_uncertainty=mean_uncertainty,
        config=config_to_dict(config),
    )

    write_json(output_dir / "config.json", config_to_dict(config))
    write_json(output_dir / "summary.json", result.to_dict())
    return result
