"""JAX implementation of ROC AUC score."""

from __future__ import annotations

import jax

from probly.metrics import auc, roc_auc_score, roc_curve


@roc_auc_score.register(jax.Array)
def roc_auc_score_jax(y_true: jax.Array, y_score: jax.Array) -> jax.Array:
    """Compute area under the ROC curve for JAX arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)
