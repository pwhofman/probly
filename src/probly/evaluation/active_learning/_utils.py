"""Utility helpers for active learning evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, cast

import numpy as np

import probly.quantification.classification as qc_cls
import probly.quantification.regression as qc_reg
from probly.representation import Sampler

type MetricFn = Callable[[np.ndarray, np.ndarray], float]
type QueryFn = Callable[[np.ndarray], np.ndarray]

_METRIC_NAMES = Literal["mse", "mae", "accuracy", "auc"]


class Estimator(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def to_numpy(x: object) -> np.ndarray:
    """Convert a torch tensor or any array-like to a numpy array."""
    try:
        import torch  # noqa: PLC0415

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def resolve_metric(metric: str | MetricFn | None) -> tuple[MetricFn, str | None]:
    """Return ``(metric_fn, metric_name)`` where *metric_fn* is higher-is-better.

    For error metrics (``"mse"``, ``"mae"``) the function returns the negated
    value so that a higher score always means better performance.

    Args:
        metric: One of ``"mse"``, ``"mae"``, ``"accuracy"``, ``"auc"``, a
            callable ``(y_true, y_pred) -> float``, or ``None`` (defaults to
            negative MSE).

    Returns:
        metric_fn: Callable with signature ``(y_true, y_pred) -> float``.
        metric_name: Original string key, or ``None`` for callables.

    Raises:
        ValueError: If *metric* is an unrecognised string.
    """
    if metric is None:
        return lambda y_true, y_pred: -float(np.mean((y_pred - y_true) ** 2)), "mse"

    if callable(metric):
        return cast("MetricFn", metric), None
    if metric == "mse":
        return lambda y_true, y_pred: -float(np.mean((y_pred - y_true) ** 2)), "mse"
    if metric == "mae":
        return lambda y_true, y_pred: -float(np.mean(np.abs(y_pred - y_true))), "mae"
    if metric == "accuracy":
        return lambda y_true, y_pred: float(np.mean(y_pred == y_true)), "accuracy"
    if metric == "auc":
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415

        return lambda y_true, y_pred: float(roc_auc_score(y_true, y_pred)), "auc"
    msg = f"Unknown metric {metric!r}. Choose from 'mse', 'mae', 'accuracy', 'auc', or pass a callable."
    raise ValueError(msg)


def default_query_fn(model: Estimator) -> QueryFn:
    """Return a default uncertainty query function inferred from *model*.

    Uses :func:`~probly.quantification.classification.margin_sampling` for
    models that expose ``predict_proba``, and
    :func:`~probly.quantification.regression.variance_conditional_expectation`
    otherwise.
    """
    if hasattr(model, "predict_proba"):
        return qc_cls.margin_sampling
    return qc_reg.variance_conditional_expectation


def get_outputs(model: Estimator, x: np.ndarray, num_samples: int) -> np.ndarray:
    """Return model outputs shaped ``(n_instances, n_samples, n_outputs)``.

    Dispatches to :func:`get_probs_classification` for models with
    ``predict_proba`` and to :func:`get_preds_regression` otherwise.
    """
    if hasattr(model, "predict_proba"):
        return get_probs_classification(model, x, num_samples)
    return get_preds_regression(model, x, num_samples)


def get_probs_classification(model: Estimator, x: np.ndarray, num_samples: int) -> np.ndarray:
    """Return class probabilities shaped ``(n_instances, n_samples, n_classes)``."""
    if num_samples == 1:
        probs = model.predict_proba(x)
        return probs[:, np.newaxis, :]
    sampler = Sampler(model.predict_proba, num_samples=num_samples)
    return np.array(sampler.predict(x))


def get_preds_regression(model: Estimator, x: np.ndarray, num_samples: int) -> np.ndarray:
    """Return predictions shaped ``(n_instances, n_samples, n_outputs)``.

    For standard regressors ``n_outputs == 1``.  For models that expose
    predictive uncertainty (e.g. evidential regression) ``n_outputs == 2``,
    where ``[:,: ,0]`` is the mean and ``[:, :, 1]`` is the variance.
    """
    if num_samples == 1:
        preds = model.predict(x)
        return preds[:, np.newaxis, np.newaxis]
    sampler = Sampler(model.predict, num_samples=num_samples)
    preds = np.array(sampler.predict(x))
    if preds.ndim == 2:
        preds = preds[:, :, np.newaxis]
    return preds


def score(
    model: Estimator,
    x: np.ndarray,
    y: np.ndarray,
    metric_fn: MetricFn,
    metric_name: str | None,
) -> float:
    """Evaluate *metric_fn* on ``(x, y)`` using *model*.

    For the ``"auc"`` metric the model's ``predict_proba`` output is used
    (binary: positive-class column; multiclass: full probability matrix).
    """
    if metric_name == "auc":
        if not hasattr(model, "predict_proba"):
            msg = "AUC metric requires a model with predict_proba."
            raise AttributeError(msg)
        proba = model.predict_proba(x)
        y_score = proba[:, 1] if proba.shape[1] == 2 else proba
        return metric_fn(y, y_score)
    return metric_fn(y, model.predict(x))


def compute_normalized_auc(scores: list[float]) -> float:
    r"""Compute the area under the score curve, normalised by the ideal AUC.

    The ideal active learning strategy always achieves a score of 1, so its
    AUC equals the length of the x-axis (number of iterations minus one).
    The normalised AUC measures what fraction of that ideal area was realised:

    .. math::

        \\text{NAUC} = \\frac{\\int s(t)\\,dt}{\\int 1\\,dt}

    where the integrals are approximated with the trapezoid rule over iteration
    indices and the denominator equals ``n - 1`` for ``n`` valid iterations.
    Scores are assumed to lie in ``[0, 1]``; ``1`` means the model achieved a
    perfect score throughout all iterations.

    ``NaN`` entries (empty pool) are excluded while preserving their original
    iteration indices so that the x-axis spacing remains correct.

    Args:
        scores: Per-iteration scores in ``[0, 1]``, higher-is-better.

    Returns:
        Normalised AUC in ``[0, 1]``, or ``nan`` if fewer than two finite
        scores are available.
    """
    s = np.asarray(scores, dtype=float)
    x = np.arange(len(s), dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return float("nan")
    s_valid = s[mask]
    x_valid = x[mask]
    actual_auc = float(np.trapezoid(s_valid, x=x_valid))
    ideal_auc = float(x_valid[-1] - x_valid[0])
    return actual_auc / ideal_auc
