"""Pool-based active learning evaluation using probly uncertainty quantification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np

from probly.evaluation.active_learning._utils import (
    Estimator,
    MetricFn,
    QueryFn,
    _build_probly_estimator,
    compute_normalized_auc,
    default_query_fn,
    get_uncertainty,
    resolve_metric,
    score,
    to_numpy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.representer._representer import Representer


# ---------------------------------------------------------------------------
# Public overloads (for type checkers and IDE completion)
# ---------------------------------------------------------------------------


@overload
def active_learning_loop(
    model: Estimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    query_fn: QueryFn | None = ...,
    metric: MetricFn | str | None = ...,
    pool_size: int = ...,
    num_samples: int = ...,
    n_iterations: int = ...,
    seed: int | None = ...,
    n_epochs: int = ...,
    batch_size: int = ...,
    pred_batch_size: int = ...,
    lr: float = ...,
    device: object = ...,
    reset_fn: Callable | str | None = ...,
) -> tuple[np.ndarray, np.ndarray, list[float], float]: ...


@overload
def active_learning_loop(
    representer: Representer,
    quantifier_fn: Callable,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    metric: MetricFn | str | None = ...,
    pool_size: int = ...,
    n_iterations: int = ...,
    seed: int | None = ...,
    n_epochs: int = ...,
    batch_size: int = ...,
    pred_batch_size: int = ...,
    lr: float = ...,
    device: object = ...,
    reset_fn: Callable | str | None = ...,
) -> tuple[np.ndarray, np.ndarray, list[float], float]: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def active_learning_loop(  # type: ignore[misc]
    *args: object,
    query_fn: QueryFn | None = None,
    metric: MetricFn | str | None = None,
    pool_size: int = 10,
    num_samples: int = 1,
    n_iterations: int = 20,
    seed: int | None = None,
    n_epochs: int = 10,
    batch_size: int = 64,
    pred_batch_size: int = 512,
    lr: float = 1e-3,
    device: object = None,
    reset_fn: Callable | Literal["default"] | None = "default",
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """Run a pool-based active learning loop and evaluate on a held-out test set.

    Supports two calling conventions:

    **Estimator interface** (original)::

        active_learning_loop(model, x_train, y_train, x_test, y_test, ...)

    **Probly representer + quantifier interface**::

        active_learning_loop(representer, quantifier_fn, x_train, y_train, x_test, y_test, ...)

    In the second form the loop builds an internal estimator from the representer
    (detecting the predictor type to know how to train) and uses
    ``representer`` + ``quantifier_fn`` for uncertainty scoring.

    The initial labeled set is drawn randomly (``pool_size`` samples) from
    *x_train* / *y_train*; the remaining training samples form the unlabeled
    pool.  At each iteration the model is retrained on the growing labeled set,
    uncertainty is scored on the pool, and the ``pool_size`` most uncertain
    samples are queried and moved to the labeled set.  Performance is measured
    on the fixed *x_test* / *y_test* split.

    Args:
        *args: Positional arguments — either ``(model, x_train, y_train, x_test, y_test)``
            or ``(representer, quantifier_fn, x_train, y_train, x_test, y_test)``.
            See the individual parameter descriptions below.
        _arg1: An sklearn-compatible estimator (``fit`` / ``predict``) **or** a
            probly :class:`~probly.representer._representer.Representer`.
        _arg2: ``x_train`` (estimator interface) **or** a quantifier callable
            such as
            :func:`~probly.quantification.measure.distribution.mutual_information`
            (representer interface).
        _arg3: ``y_train`` (estimator) or ``x_train`` (representer).
        _arg4: ``x_test`` (estimator) or ``y_train`` (representer).
        _arg5: ``y_test`` (estimator) or ``x_test`` (representer).
        _arg6: ``None`` (estimator) or ``y_test`` (representer).
        query_fn: Custom uncertainty scoring function for the estimator
            interface.  Signature ``(outputs) -> scores`` where *outputs* has
            shape ``(n_instances, n_samples, n_outputs)``.  Defaults to margin
            sampling for classifiers and variance of the conditional expectation
            for regressors.  Ignored in the representer interface.
        metric: Performance metric evaluated on the test set each iteration.
            One of ``"mse"``, ``"mae"``, ``"accuracy"``, ``"auc"`` or a
            callable.  Error metrics are negated so that higher is always
            better.  Defaults to negative MSE (accuracy if model has
            ``predict_proba``).
        pool_size: Initial labeled-set size and number of samples queried per
            iteration.
        num_samples: Stochastic forward passes for MC sampling (estimator
            interface only).
        n_iterations: Maximum active learning iterations.
        seed: Random seed for reproducible initial set selection.
        n_epochs: Training epochs per ``fit`` call (representer interface /
            torch backends only).
        batch_size: Mini-batch size during training (torch only).
        pred_batch_size: Batch size during inference (torch only).
        lr: Learning rate for the Adam optimiser (torch only).
        device: Torch device.  Auto-detected when ``None`` (torch only).
        reset_fn: Weight reset strategy before each ``fit`` call (torch only).
            ``"default"`` resets every layer that exposes
            ``reset_parameters``.  ``None`` warm-starts.  A callable resets
            in-place.

    Returns:
        x_labeled: Final labeled features.
        y_labeled: Corresponding labels.
        scores: Per-iteration test-set performance (higher-is-better).
        normalized_auc: Normalised AUC of *scores*.
    """
    # ------------------------------------------------------------------
    # Interface detection
    # ------------------------------------------------------------------
    if len(args) == 6 and callable(args[1]):
        rep, quantifier_fn_, x_train, y_train, x_test, y_test = args
        model: Estimator = cast(
            "Estimator",
            _build_probly_estimator(
                cast("Representer", rep),
                cast("Callable", quantifier_fn_),
                n_epochs=n_epochs,
                batch_size=batch_size,
                pred_batch_size=pred_batch_size,
                lr=lr,
                device=device,
                reset_fn=reset_fn,
            ),
        )
    elif len(args) == 5:
        model = cast("Estimator", args[0])
        _, x_train, y_train, x_test, y_test = args
    else:
        msg = (
            "active_learning_loop expects either 5 positional args "
            "(model, x_train, y_train, x_test, y_test) or 6 positional args "
            "(representer, quantifier_fn, x_train, y_train, x_test, y_test)."
        )
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    x_train = to_numpy(x_train)
    y_train = to_numpy(y_train)
    x_test = to_numpy(x_test)
    y_test = to_numpy(y_test)

    if metric is None and hasattr(model, "predict_proba"):
        metric = "accuracy"
    metric_fn, metric_name = resolve_metric(metric)

    if query_fn is None:
        query_fn = default_query_fn(model)

    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(x_train), size=min(pool_size, len(x_train)), replace=False)
    pool_mask = np.ones(len(x_train), dtype=bool)
    pool_mask[initial_idx] = False

    x_labeled = x_train[initial_idx].copy()
    y_labeled = y_train[initial_idx].copy()
    scores: list[float] = []

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    for _ in range(n_iterations):
        model.fit(x_labeled, y_labeled)
        scores.append(score(model, x_test, y_test, metric_fn, metric_name))

        if not pool_mask.any():
            break

        pool_indices = np.where(pool_mask)[0]
        uncertainty = get_uncertainty(model, x_train[pool_indices], num_samples, query_fn)

        n_query = min(pool_size, len(pool_indices))
        tiebreak = rng.random(len(uncertainty))
        top_positions = np.lexsort((tiebreak, uncertainty))[-n_query:]
        queried = pool_indices[top_positions]

        x_labeled = np.concatenate([x_labeled, x_train[queried]])
        y_labeled = np.concatenate([y_labeled, y_train[queried]])
        pool_mask[queried] = False

    return x_labeled, y_labeled, scores, compute_normalized_auc(scores)
