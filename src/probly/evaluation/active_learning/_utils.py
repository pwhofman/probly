"""Utility helpers for active learning evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

import numpy as np

from probly.representer import Sampler

if TYPE_CHECKING:
    import torch
    from torch import nn

    from probly.representer._representer import Representer

type MetricFn = Callable[[np.ndarray, np.ndarray], float]
type QueryFn = Callable[[np.ndarray], np.ndarray]

_METRIC_NAMES = Literal["mse", "mae", "accuracy", "auc"]


def total_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute total entropy of the mean predictive distribution.

    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)

    Returns:
        numpy.ndarray of shape (n_instances,) — entropy values, higher is
        more uncertain.
    """
    from scipy.stats import entropy as scipy_entropy  # noqa: PLC0415

    mean_probs = probs.mean(axis=1)  # (n_instances, n_classes)
    return scipy_entropy(mean_probs, axis=1)


def margin_sampling(probs: np.ndarray) -> np.ndarray:
    """Compute negative margin as an uncertainty measure for active learning.

    The margin is the difference between the two highest class probabilities
    averaged over samples.  A small margin means the model is uncertain between
    its top two classes.  The sign is flipped so that higher values correspond
    to higher uncertainty, matching the convention used by other query functions.

    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)

    Returns:
        numpy.ndarray of shape (n_instances,) — negative margin, higher is
        more uncertain.
    """
    mean_probs = probs.mean(axis=1)  # (n_instances, n_classes)
    sorted_probs = np.sort(mean_probs, axis=1)  # ascending
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return -margin


def variance_conditional_expectation(probs: np.ndarray) -> np.ndarray:
    """Compute variance of conditional expectation as the epistemic uncertainty.

    Assume that the input is from a distribution over parameters of
    a normal distribution. The first element of the parameter vector is the mean
    and the second element is the variance.
    The epistemic uncertainty is the variance of the mean of the samples.

    Args:
        probs: Parameter distributions of shape (n_instances, n_samples, 2).

    Returns:
        Variance of conditional expectation values of shape (n_instances,).

    """
    vce = np.var(probs[:, :, 0], axis=1)
    return vce


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
        from probly.metrics import roc_auc_score as _roc_auc_score  # noqa: PLC0415

        return lambda y_true, y_pred: float(_roc_auc_score(y_true, y_pred)), "auc"  # ty:ignore[invalid-argument-type]
    msg = f"Unknown metric {metric!r}. Choose from 'mse', 'mae', 'accuracy', 'auc', or pass a callable."
    raise ValueError(msg)


def default_query_fn(model: Estimator) -> QueryFn:
    """Return a default uncertainty query function inferred from *model*.

    Uses :func:`~probly.quantification.classification.margin` for
    models that expose ``predict_proba``, and
    :func:`~probly.quantification.regression.variance_conditional_expectation`
    otherwise.
    """
    if hasattr(model, "predict_proba"):
        return margin_sampling
    return variance_conditional_expectation


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


def get_uncertainty(
    model: Estimator,
    x: np.ndarray,
    num_samples: int,
    query_fn: QueryFn | None,
) -> np.ndarray:
    """Return per-instance uncertainty scores, dispatching on model type.

    If *model* exposes an ``uncertainty_scores`` method (e.g.
    :class:`_ProblyEstimator`) that method is called directly.  Otherwise the
    standard path is used: :func:`get_outputs` collects the outputs and
    *query_fn* scores them.

    Args:
        model: A fitted estimator.
        x: Pool feature array of shape ``(n_pool, n_features)``.
        num_samples: Number of stochastic forward passes (passed to
            :func:`get_outputs` on the standard path).
        query_fn: Scoring function for the standard path.  Ignored when the
            model provides ``uncertainty_scores``.

    Returns:
        Uncertainty scores of shape ``(n_pool,)``.
    """
    if hasattr(model, "uncertainty_scores"):
        return cast("Any", model).uncertainty_scores(x)
    outputs = get_outputs(model, x, num_samples)
    if query_fn is None:
        msg = "query_fn must be provided when model has no uncertainty_scores method"
        raise TypeError(msg)
    return query_fn(outputs)


# ---------------------------------------------------------------------------
# Internal probly estimator (built inside active_learning_loop)
# ---------------------------------------------------------------------------


def _default_reset(model: object) -> None:
    """Reset all layers that expose ``reset_parameters``."""
    for m in getattr(model, "modules", lambda: [model])():
        reset = getattr(m, "reset_parameters", None)
        if callable(reset):
            reset()


def _train_torch_members(
    members: nn.ModuleList,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device | str | int | None,
    reset_fn: Callable | None,
) -> None:
    """Train each member of a ``nn.ModuleList`` ensemble independently."""
    import torch  # noqa: PLC0415

    loss_fn = torch.nn.CrossEntropyLoss()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(x_t)
    for member in members:
        if reset_fn is not None:
            reset_fn(member)
        member.train()
        opt = torch.optim.Adam(member.parameters(), lr=lr)
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                out = member(x_t[idx])
                loss = loss_fn(out, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()


def _train_torch_single(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device | str | int | None,
    reset_fn: Callable | None,
) -> None:
    """Train a single ``nn.Module``."""
    import torch  # noqa: PLC0415

    loss_fn = torch.nn.CrossEntropyLoss()
    if reset_fn is not None:
        reset_fn(model)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # type: ignore[union-attr]
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(x_t)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            out = model(x_t[idx])  # type: ignore[operator]
            loss = loss_fn(out, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


class _ProblyEstimator:
    """Sklearn-compatible estimator built internally from a probly representer and quantifier.

    This class is constructed automatically inside
    :func:`~probly.evaluation.active_learning.active_learning_loop` when the
    function is called with the ``(representer, quantifier, x_train, ...)``
    signature.  It should not be instantiated directly.

    Training, prediction, and uncertainty scoring all delegate to the predictor
    wrapped by *representer*:

    - ``nn.ModuleList`` (deep ensemble / credal ensembling): each member is
      trained independently.
    - ``nn.Module`` (single model): trained as a standard torch classifier.
    - sklearn ``BaseEstimator``: ``predictor.fit(x, y)`` is called.

    For uncertainty, the torch ensemble path builds a
    ``TorchCategoricalDistributionSample`` from per-member softmax outputs and
    optionally applies the credal-set filter before calling *quantifier_fn*.
    All other paths call ``representer.predict(x)`` and pass the result
    directly to *quantifier_fn*.

    Args:
        representer: A pre-built probly representer wrapping a trainable
            predictor.
        quantifier_fn: A probly distribution measure compatible with the
            representer's output representation.
        n_epochs: Training epochs per ``fit`` call.
        batch_size: Mini-batch size during training (torch only).
        pred_batch_size: Batch size during inference (torch only).
        lr: Learning rate for the Adam optimiser (torch only).
        device: Torch device.  Auto-detected when ``None`` (torch only).
        reset_fn: Weight reset strategy before each ``fit`` call (torch only).
            ``"default"`` resets every layer that has ``reset_parameters``.
            ``None`` keeps current weights.  A callable resets in-place.
    """

    def __init__(
        self,
        representer: Representer,
        quantifier_fn: Callable,
        *,
        n_epochs: int = 10,
        batch_size: int = 64,
        pred_batch_size: int = 512,
        lr: float = 1e-3,
        device: object = None,
        reset_fn: Callable | Literal["default"] | None = "default",
    ) -> None:
        self.representer = representer
        self.quantifier_fn = quantifier_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.lr = lr

        if reset_fn == "default":
            self.reset_fn: Callable | None = _default_reset
        else:
            self.reset_fn = reset_fn  # type: ignore[assignment]

        predictor = representer.predictor
        self._backend: str

        try:
            import torch  # noqa: PLC0415
            from torch import nn  # noqa: PLC0415

            if isinstance(predictor, nn.ModuleList):
                self._backend = "torch_ensemble"
                self.device: torch.device = (
                    cast("torch.device", device)
                    if device is not None
                    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                )
                for m in predictor:
                    m.to(self.device)
                self.predict_proba = self._predict_proba  # type: ignore[method-assign]
                return
            if isinstance(predictor, nn.Module):
                self._backend = "torch_single"
                self.device = (
                    cast("torch.device", device)
                    if device is not None
                    else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                )
                predictor.to(self.device)
                self.predict_proba = self._predict_proba  # type: ignore[method-assign]
                return
        except ImportError:
            pass

        if hasattr(predictor, "fit"):
            self._backend = "sklearn"
            if hasattr(predictor, "predict_proba"):
                self.predict_proba = self._predict_proba  # type: ignore[method-assign]
            return

        msg = (
            f"Cannot determine training backend from predictor type {type(predictor).__name__}. "
            "Supported: nn.ModuleList, nn.Module, or any object with a fit() method."
        )
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # fit / predict / predict_proba
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Train the underlying model on ``(x, y)``.

        Args:
            x: Feature array of shape ``(n_samples, n_features)``.
            y: Integer class labels of shape ``(n_samples,)``.

        Returns:
            ``self``
        """
        predictor = self.representer.predictor
        if self._backend == "sklearn":
            cast("Any", predictor).fit(x, y)
        elif self._backend == "torch_ensemble":
            _train_torch_members(
                cast("nn.ModuleList", predictor),
                x,
                y,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                reset_fn=self.reset_fn,
            )
        elif self._backend == "torch_single":
            _train_torch_single(
                cast("nn.Module", predictor),
                x,
                y,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                reset_fn=self.reset_fn,
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class predictions as a numpy array.

        Args:
            x: Feature array of shape ``(n_samples, n_features)``.

        Returns:
            Integer class labels of shape ``(n_samples,)``.
        """
        if self._backend == "sklearn":
            return cast("Any", self.representer.predictor).predict(x)
        return self._predict_proba(x).argmax(axis=-1)

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_classes)``.

        Args:
            x: Feature array of shape ``(n_samples, n_features)``.

        Returns:
            Probability array.
        """
        if self._backend == "sklearn":
            return cast("Any", self.representer.predictor).predict_proba(x)
        if self._backend == "torch_ensemble":
            return self._ensemble_softmax(x).mean(axis=1)
        return self._single_softmax(x)

    # ------------------------------------------------------------------
    # uncertainty_scores
    # ------------------------------------------------------------------

    def uncertainty_scores(self, x: np.ndarray) -> np.ndarray:
        """Return per-instance uncertainty scores via the representer and quantifier.

        Args:
            x: Feature array of shape ``(n_samples, n_features)``.

        Returns:
            Uncertainty scores of shape ``(n_samples,)``.
        """
        if self._backend == "torch_ensemble":
            return self._uncertainty_torch_ensemble(x)
        if self._backend == "torch_single":
            return self._uncertainty_torch_single(x)
        return self._uncertainty_sklearn(x)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensemble_softmax(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities of shape ``(n_samples, n_members, n_classes)``."""
        import torch  # noqa: PLC0415

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        member_probs = []
        for member in cast("Any", self.representer.predictor):
            member.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(x_t), self.pred_batch_size):
                    batch = x_t[start : start + self.pred_batch_size]
                    out = torch.softmax(member(batch), dim=-1)
                    parts.append(out.cpu().numpy())
            member_probs.append(np.concatenate(parts, axis=0))
        return np.stack(member_probs, axis=1)

    def _single_softmax(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities of shape ``(n_samples, n_classes)``."""
        import torch  # noqa: PLC0415

        predictor = cast("Any", self.representer.predictor)
        predictor.eval()
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        parts = []
        with torch.no_grad():
            for start in range(0, len(x_t), self.pred_batch_size):
                batch = x_t[start : start + self.pred_batch_size]
                out = torch.softmax(predictor(batch), dim=-1)
                parts.append(out.cpu().numpy())
        return np.concatenate(parts, axis=0)

    def _uncertainty_torch_ensemble(self, x: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchCategoricalDistribution,
            TorchCategoricalDistributionSample,
        )
        from probly.representer import CredalEnsemblingRepresenter  # noqa: PLC0415

        stacked = torch.as_tensor(self._ensemble_softmax(x), dtype=torch.float32, device=self.device)
        tcd = TorchCategoricalDistribution(stacked)
        sample = TorchCategoricalDistributionSample(tensor=tcd, sample_dim=1)

        if isinstance(self.representer, CredalEnsemblingRepresenter) and self.representer.alpha > 0.0:
            from probly.representer.credal_ensembler._common import (  # noqa: PLC0415
                compute_representative_set,
            )

            filtered = cast(
                "TorchCategoricalDistributionSample",
                compute_representative_set(sample, alpha=self.representer.alpha, distance=self.representer.distance),
            )
            sample = TorchCategoricalDistributionSample(
                tensor=filtered.tensor,  # type: ignore[arg-type]
                sample_dim=filtered.sample_dim,
            )

        scores = self.quantifier_fn(sample)
        if isinstance(scores, torch.Tensor):
            return scores.detach().cpu().numpy()
        return np.asarray(scores)

    def _uncertainty_torch_single(self, x: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        rep = self.representer.predict(x_t)
        scores = self.quantifier_fn(rep)
        if isinstance(scores, torch.Tensor):
            return scores.detach().cpu().numpy()
        return np.asarray(scores)

    def _uncertainty_sklearn(self, x: np.ndarray) -> np.ndarray:
        rep = self.representer.predict(x)
        return np.asarray(self.quantifier_fn(rep))


def _build_probly_estimator(
    representer: Representer,
    quantifier_fn: Callable,
    *,
    n_epochs: int,
    batch_size: int,
    pred_batch_size: int,
    lr: float,
    device: object,
    reset_fn: Callable | Literal["default"] | None,
) -> _ProblyEstimator:
    """Build a :class:`_ProblyEstimator` from a representer and quantifier.

    Args:
        representer: Pre-built probly representer.
        quantifier_fn: Compatible distribution measure.
        n_epochs: Training epochs per fit call.
        batch_size: Mini-batch size (torch only).
        pred_batch_size: Inference batch size (torch only).
        lr: Adam learning rate (torch only).
        device: Torch device (torch only).
        reset_fn: Weight reset strategy (torch only).

    Returns:
        A ready-to-use :class:`_ProblyEstimator`.
    """
    return _ProblyEstimator(
        representer,
        quantifier_fn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        pred_batch_size=pred_batch_size,
        lr=lr,
        device=device,
        reset_fn=reset_fn,
    )


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
