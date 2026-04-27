"""AL estimators bridging the benchmark training pipeline with probly's AL protocols."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.method.ensemble import EnsemblePredictor
from probly.predictor import RandomPredictor, predict_single
from probly.quantification import quantify
from probly.quantification.decomposition import AleatoricEpistemicDecomposition
from probly.representer import representer
from probly_benchmark import models
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.train import _training_loop, train_model
from probly_benchmark.train_funcs import train_epoch_cross_entropy, validate_cross_entropy

if TYPE_CHECKING:
    from omegaconf import DictConfig

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
        batch_size: int = 512,
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
            batch_size: Batch size used during inference to avoid OOM.
        """
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.batch_size = batch_size
        self.model: nn.Module | None = None

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
        self.model = model

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass returning averaged logits for plain and ensemble."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        if isinstance(self.model, nn.ModuleList):
            return torch.stack([m(x) for m in self.model]).mean(0)
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax of logits)."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        self.model.eval()
        parts = []
        for i in range(0, len(x), self.batch_size):
            xb = x[i : i + self.batch_size].float().to(self.device)
            parts.append(self._forward_logits(xb).argmax(-1).cpu())
        return torch.cat(parts)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        self.model.eval()
        parts = []
        for i in range(0, len(x), self.batch_size):
            xb = x[i : i + self.batch_size].float().to(self.device)
            parts.append(self._forward_logits(xb).softmax(-1).cpu())
        return torch.cat(parts)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer features for BADGE."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        self.model.eval()
        base = self.model[0] if isinstance(self.model, EnsemblePredictor) else self.model
        layer_fn = _CLASSIFICATION_LAYER.get(self.base_model_name)
        if layer_fn is None:
            msg = f"No embed layer mapping for base model {self.base_model_name!r}"
            raise ValueError(msg)
        target_layer = layer_fn(base)

        embeddings: list[torch.Tensor] = []
        hook_output: list[torch.Tensor] = []

        def _hook(_module: nn.Module, inp: tuple[torch.Tensor, ...], _out: Any) -> None:  # noqa: ANN401
            hook_output.append(inp[0].detach().cpu())

        handle = target_layer.register_forward_hook(_hook)
        try:
            for i in range(0, len(x), self.batch_size):
                hook_output.clear()
                xb = x[i : i + self.batch_size].float().to(self.device)
                base(xb)
                embeddings.append(hook_output[0])
        finally:
            handle.remove()
        return torch.cat(embeddings)


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
        num_samples: int = 50,
        uncertainty_decomposition: str = "EU",
        batch_size: int = 512,
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
            num_samples: Number of Monte Carlo samples for stochastic methods.
            uncertainty_decomposition: ``"EU"`` for epistemic (default), ``"TU"`` for total.
            batch_size: Batch size used during inference to avoid OOM.
        """
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params
        self.train_kwargs = train_kwargs
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.num_samples = num_samples
        self.uncertainty_decomposition = uncertainty_decomposition
        self.batch_size = batch_size
        self.model: nn.Module | None = None
        self._rep_kwargs: dict[str, Any] = {}

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
        self.model = model
        self._rep_kwargs = {"num_samples": self.num_samples} if isinstance(model, RandomPredictor) else {}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions via predict_single -> argmax."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        self.model.eval()
        parts = []
        for i in range(0, len(x), self.batch_size):
            xb = x[i : i + self.batch_size].float().to(self.device)
            result = predict_single(self.model, xb)
            probs = result.probabilities if hasattr(result, "probabilities") else result
            parts.append(probs.argmax(-1).cpu())
        return torch.cat(parts)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Not used by UncertaintyQuery or RandomQuery."""
        msg = "UncertaintyEstimator does not support predict_proba"
        raise NotImplementedError(msg)

    @torch.no_grad()
    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty via representer -> quantify -> decomposition."""
        if self.model is None:
            msg = "Call fit() before inference"
            raise RuntimeError(msg)
        self.model.eval()
        rep = representer(self.model, **self._rep_kwargs)
        parts = []
        for i in range(0, len(x), self.batch_size):
            xb = x[i : i + self.batch_size].float().to(self.device)
            decomposition = quantify(rep.represent(xb))
            if self.uncertainty_decomposition == "EU" and isinstance(decomposition, AleatoricEpistemicDecomposition):
                scores = decomposition.epistemic
            else:
                scores = decomposition.total  # ty: ignore[unresolved-attribute]
            parts.append(scores.detach().cpu() if isinstance(scores, torch.Tensor) else torch.as_tensor(scores))
        return torch.cat(parts)
