"""Bridge between online deep classifiers and probly's uncertainty representations.

Mirrors :mod:`river_uncertainty.representation` but for deep-learning models
instead of ARF ensembles.  Two extraction strategies are provided:

* **Deep ensemble** -- N independent :class:`OnlineClassifier` instances whose
  ``predict_proba_one`` outputs are stacked into a second-order sample.
* **MC Dropout** -- a single :class:`OnlineClassifier` whose stochastic forward
  passes (dropout active) produce the sample.

Both converge on the same probly type
(:class:`ArrayCategoricalDistributionSample`) so that downstream quantification
(entropy decomposition, zero-one decomposition, ...) works identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from river_uncertainty.deep_classifier import OnlineClassifier


@dataclass(frozen=True, slots=True)
class DeepRepresentation:
    """Empirical second-order distribution from a deep model.

    Attributes:
        sample: Stacked per-member (or per-forward-pass) categorical
            distributions with shape ``(n_members, n_classes)`` and
            ``sample_axis=0``.
        classes: Ordered class labels indexing the last axis.
    """

    sample: ArrayCategoricalDistributionSample
    classes: tuple[object, ...]

    @property
    def n_members(self) -> int:
        return self.sample.sample_size

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def bma(self) -> ArrayCategoricalDistribution:
        """Bayesian model average (unweighted mean)."""
        mean = self.sample.array.probabilities.mean(axis=0)
        return ArrayCategoricalDistribution(unnormalized_probabilities=mean)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_proba_dicts(
    proba_dicts: list[dict[Any, float]],
    classes: Sequence[object] | None,
) -> tuple[np.ndarray, tuple[object, ...]]:
    """Stack a list of ``{class: prob}`` dicts into an aligned matrix.

    Returns ``(matrix, classes)`` where *matrix* has shape
    ``(len(proba_dicts), len(classes))``.
    """
    if classes is None:
        seen: set[object] = set()
        for pd in proba_dicts:
            seen.update(pd)
        classes_list: list[object] = sorted(
            seen, key=lambda c: (not isinstance(c, (int, float)), str(c))
        )
    else:
        classes_list = list(classes)

    if not classes_list:
        classes_list = [0]

    class_idx = {c: i for i, c in enumerate(classes_list)}
    k = len(classes_list)
    n = len(proba_dicts)

    matrix = np.zeros((n, k), dtype=np.float64)
    for i, pd in enumerate(proba_dicts):
        total = sum(pd.values())
        if total <= 0:
            matrix[i, :] = 1.0 / k
            continue
        for cls, p in pd.items():
            if cls in class_idx:
                matrix[i, class_idx[cls]] = p / total

    return matrix, tuple(classes_list)


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def deep_ensemble_to_probly_sample(
    classifiers: list[OnlineClassifier],
    x: dict[str, float],
    classes: Sequence[object] | None = None,
) -> DeepRepresentation:
    """Stack per-member predictions of a deep ensemble.

    Args:
        classifiers: Independent online classifiers forming the ensemble.
        x: Feature dict.
        classes: Optional fixed class order.  If ``None``, inferred from
            the union of all members' predictions.
    """
    proba_dicts = [clf.predict_proba_one(x) for clf in classifiers]
    matrix, cls_tuple = _align_proba_dicts(proba_dicts, classes)

    dist = ArrayCategoricalDistribution(unnormalized_probabilities=matrix)
    sample = ArrayCategoricalDistributionSample(array=dist, sample_axis=0)
    return DeepRepresentation(sample=sample, classes=cls_tuple)


def mc_dropout_to_probly_sample(
    classifier: OnlineClassifier,
    x: dict[str, float],
    n_forward_passes: int = 15,
) -> DeepRepresentation:
    """Run stochastic forward passes with dropout active.

    Args:
        classifier: An :class:`OnlineClassifier` whose module contains
            dropout layers.
        x: Feature dict.
        n_forward_passes: Number of MC samples.
    """
    matrix, classes = classifier.mc_forward_passes(x, n=n_forward_passes)
    dist = ArrayCategoricalDistribution(unnormalized_probabilities=matrix)
    sample = ArrayCategoricalDistributionSample(array=dist, sample_axis=0)
    return DeepRepresentation(sample=sample, classes=tuple(classes))
