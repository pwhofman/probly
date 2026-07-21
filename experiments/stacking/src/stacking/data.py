"""Dataset contract and dispatcher for the stacking playground."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """Uniform train/calib/test dataset shape consumed by every script.

    Attributes:
        X_train: Training features. Shape ``(n_train, in_features)``.
        y_train: Training labels. Shape ``(n_train,)``.
        X_calib: Calibration features used by post-hoc layers
            (temperature scaling, conformal prediction).
        y_calib: Calibration labels.
        X_test: Test features used to compute final metrics.
        y_test: Test labels.
        in_features: Number of input features. First-class so scripts can
            wire up a model without reading ``meta``.
        num_classes: Number of classes. First-class for the same reason.
        meta: Dataset-specific extras (encoder name, soft labels, vote
            counts, class names). Scripts that know they are on a specific
            dataset look these up by string key.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    in_features: int
    num_classes: int
    meta: dict[str, Any] = field(default_factory=dict)


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Dispatch to the loader matching the given dataset name.

    Args:
        name: Either ``"two_moons"`` or ``"cifar10h"``.
        **kwargs: Forwarded to the named loader. Loader-specific kwargs
            (e.g. ``encoder=`` for ``cifar10h``) are forwarded as-is.

    Returns:
        A :class:`Dataset` populated by the named loader.

    Raises:
        ValueError: If ``name`` is not a known dataset.
    """
    if name == "two_moons":
        from stacking.datasets.two_moons import load  # noqa: PLC0415

        return load(**kwargs)
    if name == "cifar10h":
        from stacking.datasets.cifar10h_embeddings import load  # noqa: PLC0415

        return load(**kwargs)
    msg = f"unknown dataset: {name!r}"
    raise ValueError(msg)
