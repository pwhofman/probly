"""BenchmarkALEstimator: AL estimator backed by the full benchmark training pipeline."""

from __future__ import annotations

import logging
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
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
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.train import train_model

logger = logging.getLogger(__name__)


def _to_device(model: nn.Module, device: torch.device) -> None:
    """Move model (or ensemble members) to device."""
    if isinstance(model, EnsemblePredictor):
        for member in model:
            member.to(device)
    else:
        model.to(device)


def _make_train_cfg(cfg: DictConfig) -> DictConfig:
    """Build a DictConfig shaped like train.py expects from the AL config.

    The _training_loop in train.py reads cfg.epochs, cfg.optimizer,
    cfg.scheduler, cfg.early_stopping, cfg.get("grad_clip_norm"),
    and cfg.get("amp").
    """
    return OmegaConf.create(
        {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "scheduler": OmegaConf.to_container(cfg.scheduler, resolve=True),
            "early_stopping": OmegaConf.to_container(cfg.early_stopping, resolve=True),
            "grad_clip_norm": cfg.get("grad_clip_norm", None),
            "amp": False,
        }
    )


def _probabilities_from_representation(rep: Any) -> torch.Tensor:  # noqa: ANN401
    """Return per-sample class probabilities from a probly Representation.

    Dispatches by representation type:
    - ``TorchCategoricalDistribution`` -> its own ``probabilities``.
    - ``TorchSample`` whose ``tensor`` is a ``TorchCategoricalDistribution`` ->
      mean of the per-sample probabilities along ``rep.sample_dim``.
    - ``TorchProbabilityIntervalsCredalSet`` -> midpoint of lower / upper bounds.
    - ``TorchConvexCredalSet`` -> mean over the vertex axis.

    Args:
        rep: A probly Representation as returned by ``predict()`` or
            ``Sampler.represent()`` / ``IterableSampler.represent()``.

    Returns:
        Tensor of shape ``(n, n_classes)``.

    Raises:
        NotImplementedError: If ``rep`` is none of the supported types, or
            is a ``TorchSample`` whose ``tensor`` is not a
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

# Default measure per Representation type. The order is evaluated by isinstance,
# so more specific types must come before less specific ones (none here at the
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


class BenchmarkALEstimator:
    """AL estimator backed by the full benchmark training pipeline.

    Each fit() call builds a fresh model via build_model() and
    trains it with train_model() which handles method-specific
    training (evidential loss, DDU GMM fitting, credal bounds, etc.)
    via flexdispatch.

    Satisfies the Estimator protocol from
    probly.evaluation.active_learning.strategies.
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
        """Initialize the estimator.

        Args:
            method_name: UQ method name (e.g. "ensemble", "ddu").
            method_params: Parameters forwarded to build_model.
            train_kwargs: Extra keyword arguments forwarded to train_model.
            cfg: Training configuration (epochs, optimizer, scheduler, etc.).
            base_model_name: Name of the base neural network architecture.
            model_type: Predictor type (e.g. "logit_classifier").
            num_classes: Number of output classes.
            device: Torch device for training and inference.
            in_features: Number of input features for tabular base models. Forwarded
                to ``BuildContext`` so ``TabularMLP`` (and similar tabular encoders)
                receives its required ``in_features`` kwarg. Ignored by base models
                that infer their input shape (e.g. CNN backbones).
            measure: Override the default uncertainty measure name. Must be a
                key in ``_MEASURES`` (e.g. ``"entropy"``,
                ``"mutual_information"``, ``"upper_entropy"``).
            num_samples: Number of stochastic forward passes for dropout.
            pred_batch_size: Batch size used during prediction.
        """
        self.method_name = method_name
        self.method_params = method_params
        self.train_kwargs = train_kwargs
        self.train_cfg = _make_train_cfg(cfg)
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
        """Return a probly Representation for the predictor on input ``x``.

        Stochastic predictors (dropout, dropconnect, bayesian) are wrapped with
        ``Sampler`` so multiple forward passes are aggregated into a Sample.
        Ensemble predictors are wrapped with ``IterableSampler`` so per-member
        predictions are aggregated into a Sample. Everything else goes through
        ``probly.predict`` directly.

        With ``sample_axis=0`` the resulting ``TorchSample`` has the sample
        dimension at axis 0 (e.g. ``(num_samples, batch, n_classes)``), which
        matches what the downstream measure functions expect when invoked
        with the default ``sample_dim``.

        Args:
            x: Input tensor of shape ``(n, ...)``.

        Returns:
            A probly :class:`Representation`. For predictors that produce a
            ``CategoricalDistribution`` (the typical ``logit_classifier`` /
            ``probabilistic_classifier`` case) this is either a
            ``TorchCategoricalDistribution`` or a ``TorchSample`` whose
            ``tensor`` is a ``TorchCategoricalDistribution``. For credal-set
            predictors it's a ``TorchProbabilityIntervalsCredalSet`` /
            ``TorchConvexCredalSet``. The return type is intentionally broad
            here; downstream helpers narrow it via ``isinstance``.
        """
        x_d = x.to(device=self.device)
        if isinstance(self.model, RandomPredictor):
            return Sampler(self.model, num_samples=self.num_samples, sample_axis=0).represent(x_d)
        if isinstance(self.model, IterablePredictor):
            return IterableSampler(self.model, sample_axis=0).represent(x_d)
        rep = predict(self.model, x_d)
        # Reduce second-order / structured representations down to a plain
        # TorchCategoricalDistribution so downstream probability extraction and
        # probly's measure registry work without per-method special cases.
        # Posterior-network / evidential -> Dirichlet -> mean categorical.
        if isinstance(rep, torch.distributions.Dirichlet):
            alpha = rep.concentration
            return TorchCategoricalDistribution(alpha / alpha.sum(dim=-1, keepdim=True))
        # DDU -> softmax categorical (densities are dropped — the benchmark
        # uses entropy of the softmax, matching the original implementation).
        if isinstance(rep, TorchDDURepresentation):
            return rep.softmax
        return rep

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> BenchmarkALEstimator:
        """Build a fresh model and train it on (x, y).

        Uses build_model() and train_model() from the existing benchmark
        pipeline so that method-specific training logic (DDU GMM fitting,
        evidential loss, etc.) is applied automatically.
        """
        import wandb  # noqa: PLC0415

        x_t = x.to(dtype=torch.float32)
        y_t = y.to(dtype=torch.long)
        dataset = TensorDataset(x_t, y_t)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

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
        """Return class probabilities of shape (n_samples, n_classes)."""
        rep = self._representation(x)
        return _probabilities_from_representation(rep).detach().cpu()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions of shape (n_samples,)."""
        return self.predict_proba(x).argmax(dim=-1)

    @torch.no_grad()
    def _embed_one(self, m: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
        """Capture the input to the last nn.Linear via a forward pre-hook.

        Args:
            m: A torch module containing at least one ``nn.Linear``.
            x_t: Input tensor already on ``self.device``.

        Returns:
            Tensor of shape ``(len(x_t), penultimate_dim)`` on CPU.
        """
        last_linear: nn.Linear | None = None
        for module in m.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            msg = f"No nn.Linear found in model for {self.method_name}; cannot embed"
            raise RuntimeError(msg)

        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured.append(inputs[0].detach().cpu())

        handle = last_linear.register_forward_pre_hook(hook)
        try:
            m.eval()
            parts: list[torch.Tensor] = []
            for start in range(0, len(x_t), self.pred_batch_size):
                captured.clear()
                m(x_t[start : start + self.pred_batch_size])
                parts.append(captured[0])
            return torch.cat(parts)
        finally:
            handle.remove()

    @torch.no_grad()
    def _encoder_features(self, encoder: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
        """Run ``encoder`` in eval mode batched and return its output on CPU.

        Args:
            encoder: A ``nn.Module`` that accepts ``x_t`` and returns features.
            x_t: Input tensor already on ``self.device``.
        """
        encoder.eval()
        parts: list[torch.Tensor] = []
        for start in range(0, len(x_t), self.pred_batch_size):
            batch = x_t[start : start + self.pred_batch_size]
            parts.append(encoder(batch).detach().cpu())
        return torch.cat(parts)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer embeddings used by BADGE.

        Dispatch is protocol-driven (no method-name strings):

        - :class:`EnsemblePredictor` -> average ``_embed_one`` across members.
        - Predictors with an ``encoder`` attribute (DDU, posterior_network) ->
          run the encoder directly (avoids capturing flow internals).
        - Otherwise -> forward pre-hook on the last ``nn.Linear``.

        Args:
            x: Input tensor of shape ``(n, ...)``.

        Returns:
            Tensor of shape ``(n, emb_dim)`` on CPU.
        """
        x_t = x.to(device=self.device)

        if isinstance(self.model, EnsemblePredictor):
            members = list(cast("Any", self.model))
            embs = [self._embed_one(member, x_t) for member in members]
            return torch.stack(embs).mean(dim=0)

        if hasattr(self.model, "encoder"):
            return self._encoder_features(cast("Any", self.model).encoder, x_t)

        return self._embed_one(cast("Any", self.model), x_t)

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty scores of shape (n_samples,).

        Uses ``self.measure`` if set, otherwise picks the default measure
        for the Representation type returned by ``self._representation(x)``.
        Both the explicit and default measures must be keys in
        :data:`_MEASURES`.
        """
        rep = self._representation(x)
        measure_name = self.measure or _default_measure_for(rep)
        return _MEASURES[measure_name](rep).detach().cpu()
