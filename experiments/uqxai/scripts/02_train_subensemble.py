"""Train one sub-ensemble of ``mlp_small`` heads per problem.

The frozen ResNet-50 encoder runs once over the images of every ImageNet class; the heads then train
on the cached features. The ImageNet validation images of the classes are the test set.
"""

from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq
import torch
from uqxai.classes import PROBLEMS, Problem
from uqxai.data import class_file
from uqxai.model import (
    build_subensemble,
    encode_image_bytes,
    encoder_of,
    head_logits,
    load_subensemble,
    save_heads,
    train_heads,
)
from uqxai.paths import DATA_DIR, RESULT_DIR, heads_checkpoint

from probly.quantification import quantify
from probly.representation.distribution import create_categorical_distribution_from_logits
from probly.representation.sample import create_sample


def load_features(split: str, problem: Problem, encoder: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the encoder features and labels of a problem's split, encoding each class on first use."""
    features, labels = [], []
    for label, imagenet_class in enumerate(problem.imagenet_classes):
        cache = DATA_DIR / "features" / split / f"{imagenet_class}.pt"
        if not cache.exists():
            images = pq.read_table(class_file(DATA_DIR, split, imagenet_class)).column("image").to_pylist()
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encode_image_bytes(encoder, images), cache)
        features.append(torch.load(cache))
        labels.append(torch.full((len(features[-1]),), label))
    return torch.cat(features), torch.cat(labels)


def evaluate(model: torch.nn.ModuleList, features: torch.Tensor, labels: torch.Tensor, names: tuple[str, ...]) -> dict:
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
            name: correct[labels == index].float().mean().item() for index, name in enumerate(names)
        },
        "member_accuracy_mean": member_accuracy.mean().item(),
        "member_accuracy_min": member_accuracy.min().item(),
        "negative_log_likelihood": torch.nn.functional.nll_loss(mean_probabilities.log(), labels).item(),
        "member_disagreement": (logits.argmax(-1) != prediction[:, None]).float().mean().item(),
        "mean_total_uncertainty": uncertainty.total.mean().item(),
        "mean_aleatoric_uncertainty": uncertainty.aleatoric.mean().item(),
        "mean_epistemic_uncertainty": uncertainty.epistemic.mean().item(),
        "confusion_matrix_rows_true": torch.stack(
            [torch.bincount(prediction[labels == index], minlength=len(names)) for index in range(len(names))]
        ).tolist(),
    }


def main() -> None:
    """Cache the features, train the heads of every chosen problem, and report the test metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--head", default="mlp_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in args.problems:
        problem = PROBLEMS[name]
        model = build_subensemble(len(problem.classes), args.head, seed=args.seed)
        train_features, train_labels = load_features("train", problem, encoder_of(model))
        test_features, test_labels = load_features("validation", problem, encoder_of(model))
        train_heads(model, train_features, train_labels, epochs=args.epochs, seed=args.seed)
        save_heads(model, heads_checkpoint(name))
        model = load_subensemble(heads_checkpoint(name), len(problem.classes), args.head)

        metrics = {
            "problem": name,
            "classes": problem.classes,
            "head": args.head,
            "num_heads": len(model),
            "train_images": dict(zip(problem.class_names, torch.bincount(train_labels).tolist(), strict=True)),
            "test_images": dict(zip(problem.class_names, torch.bincount(test_labels).tolist(), strict=True)),
            "train": evaluate(model, train_features, train_labels, problem.class_names),
            "test": evaluate(model, test_features, test_labels, problem.class_names),
        }
        (RESULT_DIR / name / "subensemble_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(name, json.dumps(metrics["test"]))


if __name__ == "__main__":
    main()
