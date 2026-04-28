"""AL estimators bridging the benchmark training pipeline with probly's AL protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibrator import calibrate
from probly.method.calibration import platt_scaling, temperature_scaling, vector_scaling
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps
from probly.method.ensemble import EnsemblePredictor
from probly.predictor import predict_single
from probly.quantification import quantify
from probly.quantification.notion import EpistemicUncertainty, Notion, TotalUncertainty
from probly.representer import representer
from probly_benchmark import models
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.train import _training_loop, train_model
from probly_benchmark.train_funcs import train_epoch_cross_entropy, validate_cross_entropy

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

logger = logging.getLogger(__name__)

# Maps base_model_name -> the final classification nn.Linear.
# embed() hooks this layer's input to capture penultimate features.
_CLASSIFICATION_LAYER = {
    "resnet18": lambda m: m.linear,
    "lenet": lambda m: m.classifier[5],
    "tabular_mlp": lambda m: m.lin_out,
}

_CALIBRATION_METHODS = {
    "temperature": lambda model, **_kw: temperature_scaling(model),
    "platt": lambda model, **_kw: platt_scaling(model),
    "vector": lambda model, **kw: vector_scaling(model, num_classes=kw.get("num_classes")),
}

_CONFORMAL_SCORES = {
    "lac": conformal_lac,
    "aps": conformal_aps,
    "raps": conformal_raps,
}


class _NoOpRun:
    """Minimal stand-in for a wandb run that silently drops all logging."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def log(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        pass


class BaseEstimator(ABC):
    """Abstract base for AL estimators.

    Subclasses must implement ``fit`` and ``predict_proba``.
    ``predict`` defaults to ``predict_proba(x).argmax(-1)``.

    Attributes:
        cfg: Hydra DictConfig with training hyperparameters.
        base_model_name: Name of the base model architecture.
        method_name: Name of the method (e.g. ``"plain"``, ``"dropout"``, ``"lac"``).
        method_params: Method-specific hyperparameters.
        num_classes: Number of output classes.
        device: Device to train and infer on.
        in_features: Input feature dimension for tabular models; ignored for image models.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        base_model_name: str,
        method_name: str,
        method_params: dict[str, Any] | None = None,
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
    ) -> None:
        """Initialize common estimator state.

        Args:
            cfg: Hydra DictConfig with training hyperparameters.
            base_model_name: Name of the base model architecture.
            method_name: Name of the method (e.g. ``"plain"``, ``"dropout"``, ``"lac"``).
            method_params: Method-specific hyperparameters.
            num_classes: Number of output classes.
            device: Device to train and infer on.
            in_features: Input feature dimension for tabular models; ignored for image models.
        """
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params or {}
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self._model: nn.Module | None = None

    @property
    def model(self) -> nn.Module:
        """Return the trained model, raising if ``fit()`` has not been called."""
        if self._model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        return self._model

    def _train_base_model(self, train_loader: DataLoader) -> nn.Module:
        """Build and train a plain base model from scratch."""
        model = models.get_base_model(
            self.base_model_name, self.num_classes, pretrained=False, in_features=self.in_features
        )
        model.to(self.device)
        _training_loop(
            model,
            train_loader,
            None,
            self.cfg,
            self.device,
            _NoOpRun(),
            {},
            train_fn=train_epoch_cross_entropy,
            val_fn=validate_cross_entropy,
        )
        return model

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train the estimator on the labeled pool."""

    @abstractmethod
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities of shape ``(n, num_classes)``."""

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax of predict_proba)."""
        return self.predict_proba(x).argmax(-1)


class BaselineEstimator(BaseEstimator):
    """AL estimator for plain/ensemble baselines with margin/badge/random strategies.

    Satisfies the ``BadgeEstimator`` protocol from
    ``probly.evaluation.active_learning``.
    """

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Build a fresh model and train from scratch on the labeled pool."""
        cal_cfg = self.cfg.method.get("calibration")

        # Split off calibration data if calibration is configured
        if cal_cfg:
            cal_split = float(cal_cfg.get("cal_split", 0.25))
            n = len(x)
            n_cal = max(1, int(n * cal_split))
            perm = torch.randperm(n)
            cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
            x_train, y_train = x[train_idx], y[train_idx]
            x_cal, y_cal = x[cal_idx], y[cal_idx]
        else:
            x_train, y_train = x, y

        train_loader = DataLoader(
            TensorDataset(x_train.float(), y_train.long()),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        if self.method_name == "plain":
            model = self._train_base_model(train_loader)
        else:
            ctx = BuildContext(
                base_model_name=self.base_model_name,
                model_type=self.cfg.model_type,
                num_classes=self.num_classes,
                pretrained=False,
                train_loader=train_loader,
                in_features=self.in_features,
            )
            model = build_model(self.method_name, dict(self.method_params), ctx)
            model.to(self.device)
            train_model(model, train_loader, None, self.cfg, self.device, _NoOpRun(), {})

        if cal_cfg:
            cal_method_name = cal_cfg.get("method", "temperature")
            cal_factory = _CALIBRATION_METHODS[cal_method_name]
            model = cal_factory(model, num_classes=self.num_classes)
            calibrate(model, y_cal.long().to(self.device), x_cal.float().to(self.device))

        self._model = model  # ty: ignore[invalid-assignment]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via predict_single."""
        self.model.eval()
        result = predict_single(self.model, x.float().to(self.device))
        if isinstance(result, torch.Tensor):
            return result.softmax(-1).cpu()
        return result.probabilities.cpu()

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer features for BADGE."""
        self.model.eval()
        base = self.model
        if hasattr(base, "predictor"):
            base = base.predictor
        if isinstance(base, EnsemblePredictor):
            base = base[0]
        layer_fn = _CLASSIFICATION_LAYER.get(self.base_model_name)
        if layer_fn is None:
            msg = f"No embed layer mapping for base model {self.base_model_name!r}"
            raise ValueError(msg)
        target_layer = layer_fn(base)
        hook_output: list[torch.Tensor] = []

        def _hook(_module: nn.Module, inp: tuple[torch.Tensor, ...], _out: Any) -> None:  # noqa: ANN401
            hook_output.append(inp[0])

        handle = target_layer.register_forward_hook(_hook)
        try:
            base(x.float().to(self.device))  # ty: ignore[call-non-callable]
        finally:
            handle.remove()
        return hook_output[0].detach().cpu()


class UncertaintyEstimator(BaseEstimator):
    """AL estimator for probly UQ methods with uncertainty/margin/random strategies.

    Satisfies the ``UncertaintyEstimator`` protocol from
    ``probly.evaluation.active_learning``.

    Attributes:
        train_kwargs: Method-specific training kwargs forwarded to ``train_model``.
        rep_kwargs: Representer parameters from the method config (e.g. ``num_samples``).
        uncertainty_notion: Which uncertainty component to use for sample selection.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        base_model_name: str,
        method_name: str,
        method_params: dict[str, Any],
        train_kwargs: dict[str, Any],
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
        rep_kwargs: dict[str, Any] | None = None,
        uncertainty_notion: type[Notion] = EpistemicUncertainty,
    ) -> None:
        """Initialize the UncertaintyEstimator.

        Args:
            cfg: Hydra DictConfig with training hyperparameters.
            base_model_name: Name of the base model architecture.
            method_name: Name of the UQ method; must be a key in ``METHODS``.
            method_params: Method-specific hyperparameters passed to ``build_model``.
            train_kwargs: Method-specific training kwargs forwarded to ``train_model``.
            num_classes: Number of output classes.
            device: Device to train and infer on.
            in_features: Input feature dimension for tabular models; ignored for image models.
            rep_kwargs: Representer parameters from the method config (e.g. ``num_samples``).
            uncertainty_notion: Which uncertainty component to use for sample selection.
        """
        super().__init__(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=method_name,
            method_params=method_params,
            num_classes=num_classes,
            device=device,
            in_features=in_features,
        )
        self.train_kwargs = train_kwargs
        self.rep_kwargs = rep_kwargs or {}
        self.uncertainty_notion = uncertainty_notion

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Build a fresh model via build_model + train_model from scratch."""
        train_loader = DataLoader(
            TensorDataset(x.float(), y.long()),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        ctx = BuildContext(
            base_model_name=self.base_model_name,
            model_type=self.cfg.model_type,
            num_classes=self.num_classes,
            pretrained=False,
            train_loader=train_loader,
            in_features=self.in_features,
        )
        model = build_model(self.method_name, dict(self.method_params), ctx)
        model.to(self.device)
        train_model(model, train_loader, None, self.cfg, self.device, _NoOpRun(), dict(self.train_kwargs))
        self._model = model

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via representer -> canonical_element."""
        self.model.eval()
        rep = representer(self.model, **self.rep_kwargs)
        canonical: TorchCategoricalDistribution = rep.represent(x.float().to(self.device)).canonical_element
        return canonical.probabilities.cpu()

    @torch.no_grad()
    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty via representer -> quantify -> decomposition."""
        self.model.eval()
        rep = representer(self.model, **self.rep_kwargs)
        decomposition = quantify(rep.represent(x.float().to(self.device)))
        scores = decomposition[self.uncertainty_notion]  # ty: ignore[not-subscriptable]
        return scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)


class ConformalEstimator(UncertaintyEstimator):
    """AL estimator using conformal prediction set size as the uncertainty signal.

    Trains a plain base model, wraps it with a conformal predictor, and calibrates
    on a held-out split. Larger prediction sets indicate higher uncertainty.

    Attributes:
        alpha: Conformal coverage target.
        cal_split: Fraction of labeled pool reserved for conformal calibration.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        base_model_name: str,
        method_name: str = "lac",
        method_params: dict[str, Any] | None = None,
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
        alpha: float = 0.1,
        cal_split: float = 0.25,
    ) -> None:
        """Initialize the ConformalEstimator.

        Args:
            cfg: Hydra DictConfig with training hyperparameters.
            base_model_name: Name of the base model architecture.
            method_name: Nonconformity score (``"lac"``, ``"aps"``, or ``"raps"``).
            method_params: Extra kwargs for the score constructor.
            num_classes: Number of output classes.
            device: Device to train and infer on.
            in_features: Input feature dimension for tabular models.
            alpha: Conformal coverage target (e.g. 0.1 for 90% coverage).
            cal_split: Fraction of labeled pool reserved for conformal calibration.
        """
        super().__init__(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=method_name,
            method_params=method_params or {},
            train_kwargs={},
            num_classes=num_classes,
            device=device,
            in_features=in_features,
            uncertainty_notion=TotalUncertainty,
        )
        self.alpha = alpha
        self.cal_split = cal_split

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Train a plain model and calibrate a conformal wrapper on a held-out split."""
        n = len(x)
        n_cal = max(1, int(n * self.cal_split))
        perm = torch.randperm(n)
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_cal, y_cal = x[cal_idx], y[cal_idx]

        train_loader = DataLoader(
            TensorDataset(x_train.float(), y_train.long()),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        base_model = self._train_base_model(train_loader)

        score_fn = _CONFORMAL_SCORES[self.method_name]
        conformal_model = score_fn(base_model, predictor_type=self.cfg.model_type, **self.method_params)
        calibrate(conformal_model, self.alpha, y_cal.long().to(self.device), x_cal.float().to(self.device))
        self._model = conformal_model  # ty: ignore[invalid-assignment]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of base model logits)."""
        self.model.eval()
        return self.model.predictor(x.float().to(self.device)).softmax(-1).cpu()  # ty: ignore[call-non-callable]
