"""Implementation for sklearn/numpy efficient credal prediction."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize
import scipy.special
from tqdm import tqdm

from ._common import compute_efficient_credal_prediction_bounds


@compute_efficient_credal_prediction_bounds.register(np.ndarray)
def _compute_bounds_numpy(
    logits_train: np.ndarray,
    targets_train: np.ndarray,
    num_classes: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class additive logit bounds via classwise relative-likelihood optimisation.

    For each class ``k`` and each direction (min/max), find the optimal additive
    logit perturbation ``x`` on column ``k`` that keeps the mean training
    relative likelihood at least ``alpha``. The relative likelihood is
    ``exp(ll(logits + x * e_k) - ll(logits))`` where ``ll`` is the mean
    per-sample log-likelihood.

    Based on :cite:`hofmanefficient` and the reference implementation at
    https://github.com/pwhofman/efficient-credal-prediction/blob/main/models.py.

    Args:
        logits_train: Training logits, shape ``(N, num_classes)``.
        targets_train: Integer training targets, shape ``(N,)``.
        num_classes: Number of classes.
        alpha: Relative-likelihood threshold in ``[0, 1]``.

    Returns:
        Tuple ``(lower, upper)`` of ``numpy.ndarray`` with shape ``(num_classes,)``
        and dtype ``float64``. ``lower[k]`` is the most-negative logit
        perturbation on class ``k`` that keeps the relative likelihood at least
        ``alpha``; ``upper[k]`` is the most-positive.
    """
    logits_np = logits_train.astype(np.float64)
    targets_np = targets_train.astype(np.int64)

    def _mean_log_likelihood(logits: np.ndarray, targets: np.ndarray) -> float:
        log_probs = scipy.special.log_softmax(logits, axis=1)
        return float(log_probs[np.arange(len(targets)), targets].mean())

    mll = _mean_log_likelihood(logits_np, targets_np)

    lower = np.zeros(num_classes, dtype=np.float64)
    upper = np.zeros(num_classes, dtype=np.float64)

    for k in tqdm(range(num_classes), desc="Credal bounds (SciPy)"):
        for direction in (1, -1):

            def fun(x: np.ndarray, direction: int = direction) -> float:
                return float(direction * x[0])

            def const(x: np.ndarray, k: int = k) -> float:
                perturbed = logits_np.copy()
                perturbed[:, k] += x[0]
                return float(np.exp(_mean_log_likelihood(perturbed, targets_np) - mll) - alpha)

            res = scipy.optimize.minimize(
                fun,
                x0=np.array([0.0]),
                constraints=[{"type": "ineq", "fun": const}],
                bounds=[(-1e4, 1e4)],
            )

            if not res.success:
                warnings.warn(
                    f"scipy.optimize.minimize did not converge for class {k} direction {direction}: {res.message}",
                    stacklevel=2,
                )

            if direction == 1:
                lower[k] = float(res.x[0])
            else:
                upper[k] = float(res.x[0])

    return lower, upper
