from __future__ import annotations

import copy
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from entmax import entmax_bisect
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import get_model, get_model_weights


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


@dataclass(slots=True)
class ImageRecord:
    image_path: str
    fold: str
    target_probs: np.ndarray


@dataclass(slots=True)
class FoldResult:
    test_fold: str
    train_size: int
    val_size: int
    test_size: int
    member_cross_entropies: list[float]
    ensemble_cross_entropy: float
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
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "encoder_name": self.encoder_name,
            "class_names": self.class_names,
            "folds": [fold.to_dict() for fold in self.folds],
            "mean_member_cross_entropy": self.mean_member_cross_entropy,
            "mean_ensemble_cross_entropy": self.mean_ensemble_cross_entropy,
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


class SoftLabelImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: list[ImageRecord],
        transform: transforms.Compose,
    ):
        self.data_root = data_root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(self.data_root / record.image_path).convert("RGB")
        target = torch.tensor(record.target_probs, dtype=torch.float32)
        return self.transform(image), target, record.image_path


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
            n_iter=self.n_iter, # number iterations to solve for entmax normalization threshold
        )
        # The loss is derived from the Fenchel-Young loss formulation for entmax.
        # This is necessary since Entmax uses a different mapping from logits to probabilities, 
        # so it has a different matched loss.
        # This is equivalent to the standard cross-entropy loss when alpha=1 (softmax)
        omega = (1 - (probabilities**self.alpha).sum(dim=1)) / (
            self.alpha * (self.alpha - 1)
        ) # Tsallis entropy, acts as regularization to prevent solution becoming too sharp too early, rewards distributions with larger entropy
        return (omega + ((probabilities - targets) * logits).sum(dim=1)).mean()


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
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    ).strip("-")


def build_run_name(config: EntmaxImageExperimentConfig) -> str:
    mode = "finetune" if not config.freeze_encoder else "linear"
    pretrained = "pretrained" if config.pretrained else "scratch"
    parts = [
        config.encoder_name,
        "entmax",
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


def load_image_dataset(dataset_dir: Path) -> tuple[list[str], list[ImageRecord]]:
    with (dataset_dir / "annotations.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    annotation_groups = payload if isinstance(payload, list) else [payload]
    vote_counts_by_image: dict[str, Counter[str]] = defaultdict(Counter)
    class_names: set[str] = set()

    for group in annotation_groups:
        for annotation in group.get("annotations", []):
            image_path = annotation["image_path"]
            class_name = annotation["class_label"]
            vote_counts_by_image[image_path][class_name] += 1
            class_names.add(class_name)

    ordered_classes = sorted(class_names)
    records: list[ImageRecord] = []

    for image_path, class_counts in sorted(vote_counts_by_image.items()):
        total_votes = sum(class_counts.values())
        target_probs = np.array(
            [
                class_counts.get(class_name, 0) / total_votes 
                for class_name in ordered_classes # checks class_counts for each possible class
            ],
            dtype=np.float32,
        )
        records.append(
            ImageRecord(
                image_path=image_path,
                fold=extract_fold_name(image_path),
                target_probs=target_probs,
            )
        )

    return ordered_classes, records


def split_train_validation_records(
    records: list[ImageRecord],
    validation_size: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(records))
    split_index = int(len(records) * (1 - validation_size))
    shuffled_records = [records[index] for index in shuffled_indices]
    return shuffled_records[:split_index], shuffled_records[split_index:]


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
    data_root: Path,
    records: list[ImageRecord],
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = SoftLabelImageDataset(data_root, records, transform)
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
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0) # Prevent log(0)
    safe_targets = np.asarray(targets, dtype=np.float64)
    return -(safe_targets * np.log(clipped)).sum(axis=1)


def entropy_per_sample(distributions: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(distributions, dtype=np.float64), 1e-8, 1.0) # Prevent log(0)
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
) -> None:
    fieldnames = ["image_path", "fold", "cross_entropy", "target_entropy"]
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
        for images, targets, _ in loader:
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
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    num_classes: int,
    config: EntmaxImageExperimentConfig,
    seed: int,
) -> tuple[nn.Module, dict[str, list[float]]]:
    set_seed(seed)

    train_loader = make_dataloader(
        data_root=config.data_root,
        records=train_records,
        transform=build_transform(
            config.encoder_name,
            train=True,
            augmentation=config.augmentation,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"), # small optimization, can speed up data transfer to GPU
    )
    val_loader = make_dataloader(
        data_root=config.data_root,
        records=val_records,
        transform=build_transform(
            config.encoder_name,
            train=False,
            augmentation=config.augmentation,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"), # small optimization, can speed up data transfer to GPU
    )

    model = ImageEntmaxClassifier(
        encoder_name=config.encoder_name,
        num_classes=num_classes,
        pretrained=config.pretrained,
        freeze_encoder=config.freeze_encoder,
        classifier_dropout=config.classifier_dropout,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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


def predict_records(
    *,
    model: nn.Module,
    records: list[ImageRecord],
    config: EntmaxImageExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = make_dataloader(
        data_root=config.data_root,
        records=records,
        transform=build_transform(
            config.encoder_name,
            train=False,
            augmentation=config.augmentation,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_paths: list[str] = []

    model.eval()
    with torch.no_grad():
        for images, targets, image_paths in loader:
            logits = model(images.to(config.device))
            probabilities = (
                entmax_bisect(
                    logits.float(),
                    alpha=config.entmax_alpha,
                    dim=1,
                    n_iter=50,
                    ensure_sum_one=True,
                )
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            all_probabilities.append(probabilities)
            all_targets.append(targets.numpy())
            all_paths.extend(image_paths)

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_targets, axis=0),
        all_paths,
    )


def train_single_member(
    *,
    member_index: int,
    member_seed: int,
    member_dir: Path,
    test_fold: str,
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    test_records: list[ImageRecord],
    class_names: list[str],
    config: EntmaxImageExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    model, history = train_single_model(
        train_records=train_records,
        val_records=val_records,
        num_classes=len(class_names),
        config=config,
        seed=member_seed,
    )
    probabilities, targets, image_paths = predict_records(
        model=model,
        records=test_records,
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

    class_names, records = load_image_dataset(config.data_root / dataset_name)
    train_records = [record for record in records if record.fold != config.test_fold]
    test_records = [record for record in records if record.fold == config.test_fold]

    if not test_records:
        raise ValueError(
            f"Fold '{config.test_fold}' was not found in dataset '{dataset_name}'."
        )

    train_records, val_records = split_train_validation_records(
        train_records,
        validation_size=config.validation_size,
        seed=config.seed,
    )

    member_probabilities: list[np.ndarray] = []
    member_losses: list[float] = []
    member_prediction_files: list[str] = []
    model_files: list[str] = []

    for member_index in range(config.ensemble_size):
        member_seed = config.seed + member_index
        member_dir = fold_dir / f"member_{member_index:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)

        probabilities, targets, member_summary, image_paths = train_single_member(
            member_index=member_index,
            member_seed=member_seed,
            member_dir=member_dir,
            test_fold=config.test_fold,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            class_names=class_names,
            config=config,
        )
        member_probabilities.append(probabilities)
        member_losses.append(float(member_summary["cross_entropy"]))
        member_prediction_files.append(member_summary["prediction_file"])
        model_files.append(member_summary["model_file"])

    ensemble_probabilities = np.mean(np.stack(member_probabilities, axis=0), axis=0)
    ensemble_loss = mean_cross_entropy(targets, ensemble_probabilities)

    ensemble_prediction_path = fold_dir / "ensemble_predictions.csv"
    export_prediction_frame(
        output_path=ensemble_prediction_path,
        image_paths=image_paths,
        targets=targets,
        probabilities=ensemble_probabilities,
        class_names=class_names,
        fold_name=config.test_fold,
        history=None,
    )

    fold_result = FoldResult(
        test_fold=config.test_fold,
        train_size=len(train_records),
        val_size=len(val_records),
        test_size=len(test_records),
        member_cross_entropies=member_losses,
        ensemble_cross_entropy=ensemble_loss,
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
        config=config_to_dict(config),
    )

    write_json(output_dir / "config.json", config_to_dict(config))
    write_json(output_dir / "summary.json", result.to_dict())
    return result
