"""Bridge between river's online ensembles and probly's uncertainty representations.

The core object here is :class:`ARFEnsembleRepresentation`. It captures, for a
single query point, the per-tree class-probability vectors of a river
:class:`river.forest.ARFClassifier`. Conceptually, that is exactly an empirical
second-order distribution: each tree is one draw from the posterior over
"plausible categorical distributions", and the forest prediction is the mean
over that sample.

Why do we need our own thin wrapper instead of going straight to probly's
:class:`probly.representation.ArrayCategoricalDistributionSample`?

1. River produces ``dict[class, proba]`` per tree, not stacked numpy arrays.
   Some trees may not have seen every class yet (online setting), so we need
   to align to a common class index.
2. ARF supports weighted voting based on each learner's running metric. We
   want to carry those weights alongside the raw sample so that downstream
   quantifiers can choose between a weighted or unweighted second-order view.
3. The probly representation is frozen/immutable. It is more ergonomic for an
   incremental experiment to have a factory that can be re-built cheaply
   after every ``learn_one``.

Everything here is intentionally kept small: the goal of Phase 1 is to
demonstrate the shape of the integration, not to productise it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from river import forest


@dataclass(frozen=True, slots=True)
class ARFEnsembleRepresentation:
    """Empirical second-order distribution produced by an ARF ensemble.

    Attributes:
        sample: The stacked per-tree categorical distributions, shaped
            ``(n_trees, n_classes)`` with the sample axis on dimension 0.
        classes: The class labels that index the last axis of ``sample``.
        weights: Optional per-tree weights (e.g. ARF's running metric score).
            Not applied to ``sample``; kept so that weighted aggregators
            can consume them separately.
    """

    sample: ArrayCategoricalDistributionSample
    classes: tuple[object, ...]
    weights: np.ndarray | None = None

    @property
    def n_trees(self) -> int:
        return self.sample.sample_size

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def bma(self) -> ArrayCategoricalDistribution:
        """Bayesian model average: the mean categorical distribution.

        If ``weights`` is set we use a weighted mean, otherwise a plain mean
        over trees.
        """
        probs = self.sample.array.probabilities  # shape: (n_trees, n_classes)
        if self.weights is None:
            mean = probs.mean(axis=0)
        else:
            w = self.weights / self.weights.sum()
            mean = (w[:, None] * probs).sum(axis=0)
        return ArrayCategoricalDistribution(unnormalized_probabilities=mean)


def river_arf_to_probly_sample(
    arf: forest.ARFClassifier,
    x: dict[str, float],
    classes: Sequence[object] | None = None,
    *,
    use_metric_weights: bool = False,
) -> ARFEnsembleRepresentation:
    """Snapshot the per-tree predictions of ``arf`` for a single point ``x``.

    Args:
        arf: A trained (or partially trained) river ARF classifier.
        x: Feature dict to predict on.
        classes: Ordered class labels used to index the last axis. If ``None``,
            inferred as the sorted union of keys across all per-tree proba
            dicts plus the forest's own proba dict.
        use_metric_weights: If ``True``, return per-tree weights based on the
            ARF's internal running metric (``metric.get()`` per base model).
            Higher-is-better metrics are passed through unchanged; if the
            metric reports ``bigger_is_better=False`` the weights are
            inverted.

    Returns:
        An :class:`ARFEnsembleRepresentation` wrapping the per-tree
        distributions as a probly ``ArrayCategoricalDistributionSample``.
    """
    per_tree_probs: list[dict[object, float]] = [m.predict_proba_one(x) for m in arf.models]

    if classes is None:
        seen: set[object] = set()
        for pt in per_tree_probs:
            seen.update(pt)
        # Also consult the forest's aggregate proba; this makes sure we don't
        # forget a class that only appears via smoothing in the ensemble.
        seen.update(arf.predict_proba_one(x))
        classes_list = sorted(seen, key=lambda c: (not isinstance(c, (int, float)), str(c)))
    else:
        classes_list = list(classes)

    if not classes_list:
        # Cold-start: no tree has learnt anything yet. Fall back to a uniform
        # distribution over a placeholder single class to keep shapes sane.
        classes_list = [0]

    class_idx = {c: i for i, c in enumerate(classes_list)}
    k = len(classes_list)
    n = len(per_tree_probs)

    matrix = np.zeros((n, k), dtype=np.float64)
    for i, pt in enumerate(per_tree_probs):
        total = sum(pt.values())
        if total <= 0:
            # Tree has no opinion yet; encode as uniform to keep entropy well
            # defined without pretending we have information.
            matrix[i, :] = 1.0 / k
            continue
        for cls, p in pt.items():
            matrix[i, class_idx[cls]] = p / total

    weights: np.ndarray | None = None
    if use_metric_weights:
        # ARF keeps one running metric per tree in ``arf._metrics``; in recent
        # river versions this is a private attribute, but it is stable and the
        # canonical way to obtain ARF's weighted-vote weights.
        tree_metrics = getattr(arf, "_metrics", None)
        if tree_metrics is None or len(tree_metrics) != len(arf.models):
            weights = np.ones(len(arf.models), dtype=np.float64)
        else:
            metric_scores = np.asarray(
                [float(mt.get()) for mt in tree_metrics],
                dtype=np.float64,
            )
            if not tree_metrics[0].bigger_is_better:
                max_score = metric_scores.max(initial=0.0) + 1e-12
                metric_scores = max_score - metric_scores
            if metric_scores.sum() <= 0:
                metric_scores = np.ones_like(metric_scores)
            weights = metric_scores

    distribution = ArrayCategoricalDistribution(unnormalized_probabilities=matrix)
    sample = ArrayCategoricalDistributionSample(array=distribution, sample_axis=0)
    return ARFEnsembleRepresentation(sample=sample, classes=tuple(classes_list), weights=weights)
