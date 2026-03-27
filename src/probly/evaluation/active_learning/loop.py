"""Pool-based active learning evaluation using probly uncertainty quantification."""

from __future__ import annotations

import numpy as np

from probly.evaluation.active_learning._utils import (
    Estimator,
    MetricFn,
    QueryFn,
    compute_normalized_auc,
    default_query_fn,
    get_outputs,
    resolve_metric,
    score,
    to_numpy,
)


def active_learning_loop(
    model: Estimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    query_fn: QueryFn | None = None,
    metric: MetricFn | str | None = None,
    pool_size: int = 10,
    num_samples: int = 1,
    n_iterations: int = 20,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    """Run a pool-based active learning loop and evaluate on a held-out test set.

    The initial labeled set is drawn randomly (``pool_size`` samples) from
    *x_train* / *y_train*; the remaining training samples form the unlabeled
    pool. At each iteration the model is retrained on the growing labeled
    set, uncertainty is scored on the pool, and the ``pool_size`` most
    uncertain samples are queried and moved to the labeled set.  Performance
    is measured on the fixed *x_test* / *y_test* split.

    Args:
        model: A sklearn-compatible estimator with ``fit`` and ``predict``.
            Models that also expose ``predict_proba`` automatically use
            classification uncertainty measures.
        x_train: Training pool features, shape ``(n_train, n_features)``.
            Accepts numpy arrays or torch tensors.
        y_train: Training pool targets, shape ``(n_train,)``.
        x_test: Held-out test features used to evaluate performance each
            iteration.
        y_test: Held-out test targets.
        query_fn: Uncertainty scoring function with signature
            ``(outputs: np.ndarray) -> np.ndarray`` where *outputs* has shape
            ``(n_instances, n_samples, n_outputs)`` and the return value has
            shape ``(n_instances,)``.  Defaults to
            :func:`~probly.quantification.classification.margin_sampling` for
            classifiers (models with ``predict_proba``) and
            :func:`~probly.quantification.regression.variance_conditional_expectation`
            for regressors.
        metric: Performance metric evaluated on the test set each iteration.
            Accepts a string (``"mse"``, ``"mae"``, ``"accuracy"``,
            ``"auc"``) or any callable ``(y_true, y_pred) -> float``.
            Error metrics (``"mse"``, ``"mae"``) are negated so that a
            *higher* score always indicates better performance.  Defaults to
            negative MSE.
        pool_size: Number of samples in the initial labeled set and the
            number of samples queried from the pool per iteration.
        num_samples: Number of stochastic forward passes for models that
            support MC sampling via :class:`~probly.representation.Sampler`.
            Use ``1`` for deterministic models.
        n_iterations: Maximum number of active learning iterations.
        seed: Optional random seed for reproducible initial set selection and
            tie-breaking.

    Returns:
        x_labeled: Final labeled features after all queries, shape
            ``(pool_size + n_iterations * pool_size, n_features)``.
        y_labeled: Corresponding labels for the labeled set.
        scores: Per-iteration performance on *x_test* / *y_test* in
            higher-is-better convention (error metrics are negated).
        normalized_auc: Area under the score curve divided by the area of the
            ideal curve.  Equals ``1.0`` when the strategy performs at its
            best level throughout; lower values indicate slower improvement.

    """
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

    for _ in range(n_iterations):
        model.fit(x_labeled, y_labeled)
        scores.append(score(model, x_test, y_test, metric_fn, metric_name))

        if not pool_mask.any():
            break

        pool_indices = np.where(pool_mask)[0]
        outputs = get_outputs(model, x_train[pool_indices], num_samples)
        uncertainty = query_fn(outputs)

        n_query = min(pool_size, len(pool_indices))
        tiebreak = rng.random(len(uncertainty))
        top_positions = np.lexsort((tiebreak, uncertainty))[-n_query:]
        queried = pool_indices[top_positions]

        x_labeled = np.concatenate([x_labeled, x_train[queried]])
        y_labeled = np.concatenate([y_labeled, y_train[queried]])
        pool_mask[queried] = False

    return x_labeled, y_labeled, scores, compute_normalized_auc(scores)
