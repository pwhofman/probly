"""AL estimators bridging the benchmark training pipeline with probly's AL protocols."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.method.ensemble import EnsemblePredictor
from probly.quantification import quantify
from probly.quantification.notion import EpistemicUncertainty, Notion
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


class _NoOpRun:
    """Minimal stand-in for a wandb run that silently drops all logging."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def log(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        pass


class BaselineEstimator:
    """AL estimator for plain/ensemble baselines with margin/badge/random strategies.

    Satisfies the ``BadgeEstimator`` protocol from
    ``probly.evaluation.active_learning``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        base_model_name: str,
        method_name: str,
        method_params: dict[str, Any],
        num_classes: int,
        device: torch.device,
        in_features: int | None = None,
    ) -> None:
        """Initialize the BaselineEstimator.

        Args:
            cfg: Hydra DictConfig with training hyperparameters.
            base_model_name: Name of the base model architecture.
            method_name: ``"plain"`` or ``"ensemble"``.
            method_params: Method-specific hyperparameters passed to ``build_model``.
            num_classes: Number of output classes.
            device: Device to train and infer on.
            in_features: Input feature dimension for tabular models; ignored for image models.
        """
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params
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

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Build a fresh model and train from scratch on the labeled pool."""
        train_loader = DataLoader(
            TensorDataset(x.float(), y.long()),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        if self.method_name == "plain":
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
        self._model = model

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass returning averaged logits for plain and ensemble."""
        if isinstance(self.model, nn.ModuleList):
            return torch.stack([m(x) for m in self.model]).mean(0)
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax of logits)."""
        self.model.eval()
        return self._forward_logits(x.float().to(self.device)).argmax(-1).cpu()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        self.model.eval()
        return self._forward_logits(x.float().to(self.device)).softmax(-1).cpu()

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer features for BADGE."""
        self.model.eval()
        base = self.model[0] if isinstance(self.model, EnsemblePredictor) else self.model
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
            base(x.float().to(self.device))
        finally:
            handle.remove()
        return hook_output[0].detach().cpu()


class UncertaintyEstimator:
    """AL estimator for probly UQ methods with uncertainty/random strategies.

    Satisfies the ``UncertaintyEstimator`` protocol from
    ``probly.evaluation.active_learning``.
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
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params
        self.train_kwargs = train_kwargs
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.rep_kwargs = rep_kwargs or {}
        self.uncertainty_notion = uncertainty_notion
        self._model: nn.Module | None = None

    @property
    def model(self) -> nn.Module:
        """Return the trained model, raising if ``fit()`` has not been called."""
        if self._model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        return self._model

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
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax of predict_proba)."""
        return self.predict_proba(x).argmax(-1)

    @torch.no_grad()
    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty via representer -> quantify -> decomposition."""
        self.model.eval()
        rep = representer(self.model, **self.rep_kwargs)
        decomposition = quantify(rep.represent(x.float().to(self.device)))
        scores = decomposition[self.uncertainty_notion]  # ty: ignore[not-subscriptable]
        return scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)
