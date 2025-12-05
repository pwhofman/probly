"""Utility functions for evaluating OOD performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import auc, roc_curve


@dataclass
class OodEvaluationResult:
    """Container holding OOD evaluation metrics."""

    auroc: float
    fpr: np.ndarray
    tpr: np.ndarray
    labels: np.ndarray
    preds: np.ndarray


def evaluate_ood_performance(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> OodEvaluationResult:
    """Compute OOD metrics based on ID and OOD score arrays.

    Logic:
        Low score → OOD.

    Args:
        id_scores: Confidence scores for in-distribution samples.
        ood_scores: Confidence scores for out-of-distribution samples.

    Returns:
        An `OodEvaluationResult` containing AUROC, FPR, TPR, labels, and preds.
    """
    id_scores = np.clip(id_scores, 0.0, 1.0)
    ood_scores = np.clip(ood_scores, 0.0, 1.0)

    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    all_scores = np.concatenate([id_scores, ood_scores])

    preds = 1.0 - all_scores

    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return OodEvaluationResult(
        auroc=roc_auc,
        fpr=fpr,
        tpr=tpr,
        labels=labels,
        preds=preds,
    )
