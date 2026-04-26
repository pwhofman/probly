"""UQ AL estimator for probly UQ methods.

Used for every method in :data:`probly_benchmark.builders.METHODS` plus
``ensemble`` paired with the ``uncertainty`` strategy. Builds the model via
:func:`probly_benchmark.builders.build_model`, trains it via
:func:`probly_benchmark.train.train_model` (which dispatches by predictor
type for evidential loss, DDU GMM fitting, credal bounds, and so on), and
exposes the per-sample uncertainty score that the
:class:`probly.evaluation.active_learning.strategies.UncertaintyQuery`
strategy consumes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.method.ensemble import EnsemblePredictor
from probly.predictor import IterablePredictor, RandomPredictor, predict
from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.quantification.measure.distribution import (
    entropy,
    entropy_of_expected_predictive_distribution,
    mutual_information,
)
from probly.representation.credal_set.torch import (
    TorchConvexCredalSet,
    TorchProbabilityIntervalsCredalSet,
)
from probly.representation.ddu.torch import TorchDDURepresentation
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample
from probly.representer.sampler import IterableSampler, Sampler
from probly_benchmark.al_estimator._common import make_train_cfg
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.train import train_model

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _to_device(model: nn.Module, device: torch.device) -> None:
    """Move model (or ensemble members) to ``device``."""
    if isinstance(model, EnsemblePredictor):
        for member in model:
            member.to(device)
    else:
        model.to(device)


def _probabilities_from_representation(rep: Any) -> torch.Tensor:  # noqa: ANN401
    """Return per-sample class probabilities from a probly Representation.

    Dispatches by representation type:

    - :class:`TorchCategoricalDistribution` -> its own ``probabilities``.
    - :class:`TorchSample` whose ``tensor`` is a
      :class:`TorchCategoricalDistribution` -> mean of per-sample
      probabilities along ``rep.sample_dim``.
    - :class:`TorchProbabilityIntervalsCredalSet` -> midpoint of lower and
      upper bounds.
    - :class:`TorchConvexCredalSet` -> mean over the vertex axis.

    Args:
        rep: A probly :class:`Representation` as returned by ``predict()``
            or ``Sampler.represent()`` / ``IterableSampler.represent()``.

    Returns:
        Tensor of shape ``(n, num_classes)``.

    Raises:
        NotImplementedError: If ``rep`` is none of the supported types, or
            if it is a ``TorchSample`` whose ``tensor`` is not a
            ``TorchCategoricalDistribution``.
    """
    if isinstance(rep, TorchCategoricalDistribution):
        return rep.probabilities
    if isinstance(rep, TorchSample):
        if not isinstance(rep.tensor, TorchCategoricalDistribution):
            msg = f"TorchSample with non-CategoricalDistribution tensor {type(rep.tensor).__name__} is not supported."
            raise NotImplementedError(msg)
        return rep.tensor.probabilities.mean(dim=rep.sample_dim)
    if isinstance(rep, TorchProbabilityIntervalsCredalSet):
        return (rep.lower_bounds + rep.upper_bounds) * 0.5
    if isinstance(rep, TorchConvexCredalSet):
        return rep.tensor.probabilities.mean(dim=-2)
    msg = f"No probability extraction for representation {type(rep).__name__}"
    raise NotImplementedError(msg)


_MEASURES: dict[str, Any] = {
    "entropy": entropy,
    "entropy_of_expected_predictive_distribution": entropy_of_expected_predictive_distribution,
    "mutual_information": mutual_information,
    "upper_entropy": upper_entropy,
    "lower_entropy": lower_entropy,
}

# Default measure per Representation type. Evaluated by isinstance, so more
# specific types must come before less specific ones (none here at the
# moment, but worth noting if you add subclasses later).
_DEFAULT_MEASURE_BY_REP: dict[type, str] = {
    TorchCategoricalDistribution: "entropy",
    TorchSample: "entropy_of_expected_predictive_distribution",
    TorchProbabilityIntervalsCredalSet: "upper_entropy",
    TorchConvexCredalSet: "upper_entropy",
}


def _default_measure_for(rep: Any) -> str:  # noqa: ANN401
    """Return the default measure name for a probly Representation instance."""
    for rep_type, measure_name in _DEFAULT_MEASURE_BY_REP.items():
        if isinstance(rep, rep_type):
            return measure_name
    msg = f"No default measure for representation {type(rep).__name__}"
    raise NotImplementedError(msg)


class UQALEstimator:
    """AL estimator backed by the full benchmark training pipeline.

    Each ``fit`` call builds a fresh model via
    :func:`probly_benchmark.builders.build_model` and trains it with
    :func:`probly_benchmark.train.train_model`, which dispatches by
    predictor type so method-specific training (evidential loss, DDU GMM
    fitting, credal bounds, ensemble members, ...) is applied
    automatically.

    Satisfies the :class:`Estimator` protocol from
    :mod:`probly.evaluation.active_learning.strategies` and additionally
    exposes :meth:`uncertainty_scores` for
    :class:`UncertaintyQuery`.
    """

    def __init__(
        self,
        *,
        method_name: str,
        method_params: dict[str, Any],
        train_kwargs: dict[str, Any],
        cfg: DictConfig,
        base_model_name: str,
        model_type: str,
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
        measure: str | None = None,
        num_samples: int = 10,
        pred_batch_size: int = 512,
    ) -> None:
        """Initialise the estimator.

        Args:
            method_name: probly UQ method name (e.g. ``"dropout"``,
                ``"ddu"``, ``"ensemble"``).
            method_params: Parameters forwarded to :func:`build_model`.
            train_kwargs: Extra keyword arguments forwarded to
                :func:`train_model`.
            cfg: Training configuration (epochs, optimizer, scheduler,
                early stopping, gradient clipping).
            base_model_name: Name of the base architecture.
            model_type: Predictor type (e.g. ``"logit_classifier"``).
            num_classes: Number of output classes.
            device: Torch device for training and inference.
            in_features: Input feature dimension for tabular base models.
                Forwarded to :class:`BuildContext`. ``None`` for image
                base models that infer their input shape.
            measure: Override the default uncertainty measure name. Must
                be a key in :data:`_MEASURES`.
            num_samples: Number of stochastic forward passes for
                :class:`RandomPredictor` wrappers (dropout, dropconnect,
                ...).
            pred_batch_size: Batch size used during prediction.
        """
        self.method_name = method_name
        self.method_params = method_params
        self.train_kwargs = train_kwargs
        self.train_cfg = make_train_cfg(cfg)
        self.base_model_name = base_model_name
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.num_samples = num_samples
        self.pred_batch_size = pred_batch_size
        self.batch_size = cfg.batch_size

        self.measure = measure
        self.model: nn.Module | None = None

    def _representation(self, x: torch.Tensor) -> Any:  # noqa: ANN401
        """Return a probly :class:`Representation` for ``self.model`` on ``x``.

        Stochastic predictors (dropout, dropconnect, bayesian) are wrapped
        with :class:`Sampler` so multiple forward passes aggregate into a
        :class:`TorchSample`. :class:`IterablePredictor` instances (e.g.
        ensemble) are wrapped with :class:`IterableSampler` so per-member
        predictions aggregate into a sample. Everything else goes through
        :func:`probly.predict` directly.

        Second-order or structured representations are reduced to a plain
        :class:`TorchCategoricalDistribution` so downstream probability
        extraction and measure dispatch work without per-method special
        cases:

        - Posterior-network and evidential return a
          :class:`torch.distributions.Dirichlet`; we collapse it to its
          mean categorical.
        - DDU returns a :class:`TorchDDURepresentation`; we keep only the
          softmax categorical, matching the original benchmark.
        """
        x_d = x.to(device=self.device)
        if isinstance(self.model, RandomPredictor):
            return Sampler(self.model, num_samples=self.num_samples, sample_axis=0).represent(x_d)
        if isinstance(self.model, IterablePredictor):
            return IterableSampler(self.model, sample_axis=0).represent(x_d)
        rep = predict(self.model, x_d)
        if isinstance(rep, torch.distributions.Dirichlet):
            alpha = rep.concentration
            return TorchCategoricalDistribution(alpha / alpha.sum(dim=-1, keepdim=True))
        if isinstance(rep, TorchDDURepresentation):
            return rep.softmax
        return rep

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> UQALEstimator:
        """Build a fresh model and train it on ``(x, y)``.

        Uses :func:`build_model` and :func:`train_model` so that
        method-specific training logic (DDU GMM fitting, evidential loss,
        ...) is applied automatically.
        """
        import wandb  # noqa: PLC0415

        x_t = x.to(dtype=torch.float32)
        y_t = y.to(dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=self.batch_size, shuffle=True)

        ctx = BuildContext(
            base_model_name=self.base_model_name,
            model_type=self.model_type,
            num_classes=self.num_classes,
            pretrained=False,
            train_loader=train_loader,
            in_features=self.in_features,
        )
        model = build_model(self.method_name, dict(self.method_params), ctx)

        # Workaround: EfficientCredalPredictor registers lower/upper as None
        # buffers, but train.py reads model.lower.shape[0] before the bounds
        # are computed. Pre-init zero buffers here so the read works. Should
        # be fixed inside the predictor's __init__ instead.
        if self.method_name == "efficient_credal_prediction":
            model_ = cast("Any", model)
            model_.lower = torch.zeros(self.num_classes)
            model_.upper = torch.zeros(self.num_classes)

        _to_device(model, self.device)

        # Disabled wandb run for per-iteration training (no epoch-level logging).
        run = wandb.init(mode="disabled")
        train_model(
            model,
            train_loader,
            None,  # no validation
            self.train_cfg,
            self.device,
            run,
            dict(self.train_kwargs),
        )
        run.finish()

        self.model = model
        return self

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities of shape ``(n, num_classes)`` on CPU."""
        rep = self._representation(x)
        return _probabilities_from_representation(rep).detach().cpu()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions of shape ``(n,)`` on CPU."""
        return self.predict_proba(x).argmax(dim=-1)

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty scores of shape ``(n,)`` on CPU.

        Uses ``self.measure`` if set, otherwise picks the default measure
        for the representation type returned by
        :meth:`_representation`. Both the explicit and default measures
        must be keys in :data:`_MEASURES`.
        """
        rep = self._representation(x)
        measure_name = self.measure or _default_measure_for(rep)
        return _MEASURES[measure_name](rep).detach().cpu()
