"""Train the sub-ensemble of ``mlp_small`` heads on the three-class ImageNet subset.

The frozen ResNet-50 encoder runs once over the images; the heads then train on the cached features.
The official ImageNet validation images of the three groups are the test set.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from uqxai.classes import CLASS_NAMES
from uqxai.model import build_subensemble, encode_image_bytes, encoder_of, head_logits, save_heads, train_heads
from uqxai.paths import DATA_DIR, HEADS_CHECKPOINT, RESULT_DIR

from probly.quantification import quantify
from probly.representation.distribution import create_categorical_distribution_from_logits
from probly.representation.sample import create_sample


def load_features(split: str, model: torch.nn.ModuleList) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the cached encoder features and labels of a split, computing them on first use."""
    cache = DATA_DIR / f"features_{split}.pt"
    if cache.exists():
        cached = torch.load(cache)
        return cached["features"], cached["labels"]
    table = pq.read_table(DATA_DIR / f"{split}.parquet")
    labels = torch.from_numpy(table.column("label").to_numpy().astype(np.int64))
    features = encode_image_bytes(encoder_of(model), table.column("image").to_pylist())
    torch.save({"features": features, "labels": labels}, cache)
    return features, labels


def evaluate(model: torch.nn.ModuleList, features: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    """Compute accuracy and uncertainty statistics of the sub-ensemble on cached features."""
    logits = head_logits(model, features)  # (m, heads, classes)
    member_accuracy = (logits.argmax(-1) == labels[:, None]).float().mean(0)
    mean_probabilities = logits.softmax(-1).mean(1)
    prediction = mean_probabilities.argmax(-1)
    correct = prediction == labels
    # The second-order representation of probly: a sample of categorical distributions, one per head.
    sample = create_sample([create_categorical_distribution_from_logits(logits[:, h]) for h in range(logits.shape[1])])
    uncertainty = quantify(sample)
    return {
        "accuracy": correct.float().mean().item(),
        "accuracy_per_class": {
            name: correct[labels == index].float().mean().item() for index, name in enumerate(CLASS_NAMES)
        },
        "member_accuracy_mean": member_accuracy.mean().item(),
        "member_accuracy_min": member_accuracy.min().item(),
        "negative_log_likelihood": torch.nn.functional.nll_loss(mean_probabilities.log(), labels).item(),
        "member_disagreement": (logits.argmax(-1) != prediction[:, None]).float().mean().item(),
        "mean_total_uncertainty": uncertainty.total.mean().item(),
        "mean_aleatoric_uncertainty": uncertainty.aleatoric.mean().item(),
        "mean_epistemic_uncertainty": uncertainty.epistemic.mean().item(),
        "confusion_matrix_rows_true": torch.stack(
            [
                torch.bincount(prediction[labels == index], minlength=len(CLASS_NAMES))
                for index in range(len(CLASS_NAMES))
            ]
        ).tolist(),
    }


def main() -> None:
    """Cache the features, train the heads, and report the test metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="mlp_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = build_subensemble(args.head, seed=args.seed)
    train_features, train_labels = load_features("train", model)
    test_features, test_labels = load_features("validation", model)
    train_heads(model, train_features, train_labels, epochs=args.epochs, seed=args.seed)
    save_heads(model, HEADS_CHECKPOINT)

    metrics = {
        "head": args.head,
        "num_heads": len(model),
        "classes": CLASS_NAMES,
        "train_images": dict(zip(CLASS_NAMES, torch.bincount(train_labels).tolist(), strict=True)),
        "test_images": dict(zip(CLASS_NAMES, torch.bincount(test_labels).tolist(), strict=True)),
        "train": evaluate(model, train_features, train_labels),
        "test": evaluate(model, test_features, test_labels),
    }
    (RESULT_DIR / "subensemble_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
