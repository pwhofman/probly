"""Dirichlet relative likelihood non-conformity score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from flextype import flexdispatch
from probly.conformal_scores import NonConformityScore
from probly.representation.array_like import ArrayLike


@flexdispatch
def dirichlet_rl_score_func[T](alphas: T, y_true: T | None = None) -> T:
    """Compute Dirichlet relative likelihood non-conformity score.

    The score is 1 - alpha_y / max_k(alpha_k), measuring how much the
    Dirichlet concentrates on the true class relative to the most concentrated class.
    """
    msg = "Dirichlet relative likelihood score not implemented for this type."
    raise NotImplementedError(msg)


@dirichlet_rl_score_func.register(np.ndarray | ArrayLike)
def compute_dirichlet_rl_score_numpy(alphas: np.ndarray | ArrayLike, y_true: np.ndarray | ArrayLike) -> np.ndarray:
    """Compute the Dirichlet relative likelihood score using NumPy.

    Args:
        alphas: Dirichlet concentration parameters, shape (..., K).
        y_true: Ground truth class labels, shape (...,).
    """
    alphas_np = np.asarray(alphas)
    y_true_np = np.asarray(y_true).astype(int)
    alpha_y = np.take_along_axis(alphas_np, y_true_np[..., np.newaxis], axis=-1).squeeze(-1)
    alpha_max = np.max(alphas_np, axis=-1)
    return 1.0 - alpha_y / alpha_max


@dataclass(frozen=True, slots=True)
class DirichletRLScore[In, Out](NonConformityScore):
    """Dirichlet relative likelihood non-conformity score."""

    def __call__(self, y_pred: In, y_true: In | None = None) -> Any:  # noqa: ANN401
        if y_true is None:
            msg = "y_true is required for Dirichlet relative likelihood score."
            raise ValueError(msg)
        return dirichlet_rl_score_func(y_pred, y_true)


dirichlet_rl_score = DirichletRLScore()

__all__ = ["DirichletRLScore", "dirichlet_rl_score", "dirichlet_rl_score_func"]
