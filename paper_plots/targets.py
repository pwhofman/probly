"""(project, dataset, base_model) tuples that fetch + plot scripts iterate over.

Edit this list to add or remove a dataset from every fetch / plot run at once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Target:
    """One (wandb project, ID dataset, base model) combination to fetch and plot."""

    project: str
    dataset: str
    base_model: str


ENTITY = "probly"

TARGETS: tuple[Target, ...] = (
    Target(project="cifar10-benchmark", dataset="cifar10", base_model="resnet18"),
    Target(project="paper-models", dataset="imagenet", base_model="resnet50"),
)
