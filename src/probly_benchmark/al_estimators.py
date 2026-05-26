"""AL estimators bridging the benchmark training pipeline with probly's AL protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibrator import calibrate
from probly.method.calibration import CalibrationPredictor, temperature_scaling, vector_scaling
from probly.method.conformal import ConformalSetPredictor, conformal_aps, conformal_lac, conformal_raps
from probly.predictor import predict
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.builders import BuildContext, build_model
from probly_benchmark.decision import decide
from probly_benchmark.train import train_model
from probly_benchmark.uncertainty import select_uncertainty

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from probly.quantification.notion import NotionName
    from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution

# Maps base_model_name -> the final classification nn.Linear.
# embed() hooks this layer's input to capture penultimate features.
_CLASSIFICATION_LAYER = {
    "resnet18": lambda m: m.linear,
    "lenet": lambda m: m.classifier[5],
    "tabular_mlp": lambda m: m.lin_out,
}

_CALIBRATION_METHODS = {
    "temperature_scaling": lambda model, **_kw: temperature_scaling(model),
    "vector_scaling": lambda model, **kw: vector_scaling(model, num_classes=kw["num_classes"]),
}

_CONFORMAL_METHODS = {
    "conformal_lac": conformal_lac,
    "conformal_aps": conformal_aps,
    "conformal_raps": conformal_raps,
}


def extract_penultimate_features(
    model: nn.Module,
    x: torch.Tensor,
    base_model_name: str,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool = False,
) -> torch.Tensor:
    """Extract penultimate-layer features from a model using a forward hook.

    Registers a hook on the final classification layer (looked up via
    ``_CLASSIFICATION_LAYER``), runs batched forward passes with
    ``torch.compile`` disabled so hooks fire reliably, and returns the
    concatenated inputs to that layer.

    Args:
        model: The base model (unwrapped, not a calibration/ensemble wrapper).
        x: Input features tensor.
        base_model_name: Key into ``_CLASSIFICATION_LAYER`` (e.g. ``"tabular_mlp"``).
        batch_size: Inference batch size.
        device: Device to run inference on.
        amp_enabled: Whether to use automatic mixed precision.

    Returns:
        Feature tensor of shape ``(n, emb_dim)`` on CPU.

    Raises:
        ValueError: If ``base_model_name`` has no entry in ``_CLASSIFICATION_LAYER``.
    """
    layer_fn = _CLASSIFICATION_LAYER.get(base_model_name)
    if layer_fn is None:
        msg = f"No embed layer mapping for base model {base_model_name!r}"
        raise ValueError(msg)
    target_layer = layer_fn(model)
    hook_output: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inp: tuple[torch.Tensor, ...], _out: Any) -> None:  # noqa: ANN401
        hook_output.append(inp[0].detach().cpu())

    # torch.compiler.disable ensures hooks fire even if model.forward was
    # compiled by _maybe_compile_forward during training.
    handle = target_layer.register_forward_hook(_hook)
    eager_forward = torch.compiler.disable(model)
    for i in range(0, len(x), batch_size):
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            eager_forward(x[i : i + batch_size].float().to(device))
    handle.remove()
    return torch.cat(hook_output)


class _NoOpRun:
    """Minimal stand-in for a wandb run that silently drops all logging."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def log(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        pass


class BaseEstimator:
    """Abstract base for AL estimators.

    Subclasses must implement ``fit`` and ``predict_proba``.
    ``predict`` defaults to ``predict_proba(x).argmax(-1)``.

    Attributes:
        cfg: Hydra DictConfig with training hyperparameters.
        base_model_name: Name of the base model architecture.
        method_name: Name of the method (e.g. ``"base"``, ``"dropout"``, ``"lac"``).
        method_params: Method-specific hyperparameters.
        num_classes: Number of output classes.
        device: Device to train and infer on.
        train_kwargs: Additional training hyperparameters.
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
        train_kwargs: dict[str, Any] | None = None,
        in_features: int | None = None,
    ) -> None:
        """Initialize common estimator state.

        Args:
            cfg: Hydra DictConfig with training hyperparameters.
            base_model_name: Name of the base model architecture.
            method_name: Name of the method (e.g. ``"base"``, ``"dropout"``, ``"lac"``).
            method_params: Method-specific hyperparameters.
            num_classes: Number of output classes.
            device: Device to train and infer on.
            train_kwargs: Additional training hyperparameters forwarded to ``train_model``.
            in_features: Input feature dimension for tabular models; ignored for image models.
        """
        self.cfg = cfg
        self.base_model_name = base_model_name
        self.method_name = method_name
        self.method_params = method_params or {}
        self.num_classes = num_classes
        self.device = device
        self.in_features = in_features
        self.train_kwargs = train_kwargs or {}
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
        needs_calibration = self.cfg.calibration.name != "none"
        needs_conformal = self.cfg.conformal.name != "none"

        if not needs_calibration and not needs_conformal:
            self._model = self.fit_model(x, y)
            return

        calibration_split = float(self.cfg.calibration_size)
        x_train, y_train, x_cal, y_cal = self.split_calibration_data(x, y, calibration_split, seed=self.cfg.seed)
        model = self.fit_model(x_train, y_train)
        if needs_calibration:
            self._model = self.calibrate(model, x_cal, y_cal)  # ty: ignore[invalid-assignment]
        else:
            self._model = self.conformal(model, x_cal, y_cal)  # ty: ignore[invalid-assignment]

    @staticmethod
    def split_calibration_data(
        x: torch.Tensor,
        y: torch.Tensor,
        calibration_split: float,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split ``(x, y)`` into training and calibration subsets."""
        n = len(x)
        n_cal = max(1, int(n * calibration_split))
        g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        perm = torch.randperm(n, generator=g)
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
        return x[train_idx], y[train_idx], x[cal_idx], y[cal_idx]

    def calibrate(self, model: nn.Module, x_cal: torch.Tensor, y_cal: torch.Tensor) -> CalibrationPredictor:
        """Wrap model with a calibration method and calibrate on held-out data."""
        calibration_method = _CALIBRATION_METHODS[self.cfg.calibration.name]
        cal_params = dict(self.cfg.calibration.get("params", {}))
        calibrated = calibration_method(model, num_classes=self.num_classes, **cal_params)
        calibrate(calibrated, y_cal.long().to(self.device), x_cal.float().to(self.device))
        return calibrated

    def conformal(self, model: nn.Module, x_cal: torch.Tensor, y_cal: torch.Tensor) -> ConformalSetPredictor:
        """Wrap model with a conformal predictor and calibrate on held-out data."""
        conformal_method = _CONFORMAL_METHODS[self.cfg.conformal.name]
        conformal_params = dict(self.cfg.conformal.get("params", {}))
        conformal_model = conformal_method(model, predictor_type=self.cfg.model_type, **conformal_params)
        calibrate(
            conformal_model, self.cfg.conformal.alpha, y_cal.long().to(self.device), x_cal.float().to(self.device)
        )
        return conformal_model

    def fit_model(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """Fit the model with ``x`` and ``y``."""
        train_loader = DataLoader(
            TensorDataset(x.float(), y.long()),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
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
        return model

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via probly's predict dispatch."""
        self.model.eval()
        amp_enabled = self.cfg.get("amp", False)
        parts = []
        for i in range(0, len(x), self.cfg.batch_size):
            xb = x[i : i + self.cfg.batch_size].float().to(self.device)
            with torch.amp.autocast(self.device.type, enabled=amp_enabled):
                out: TorchCategoricalDistribution = predict(self.model, xb)
            parts.append(out.probabilities.float().cpu())
        return torch.cat(parts)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (argmax of predict_proba)."""
        return self.predict_proba(x).argmax(-1)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer features for BADGE."""
        self.model.eval()
        base = self.model
        if isinstance(base, CalibrationPredictor):
            base = base.predictor
        return extract_penultimate_features(
            base,
            x,
            self.base_model_name,
            self.cfg.batch_size,
            self.device,
            amp_enabled=self.cfg.get("amp", False),
        )


class UncertaintyEstimator(BaseEstimator):
    """AL estimator for probly UQ methods with uncertainty/margin/random strategies.

    Satisfies the ``UncertaintyEstimator`` protocol from
    ``probly.evaluation.active_learning``.

    Attributes:
        rep_kwargs: Representer parameters from the method config (e.g. ``num_samples``).
        uncertainty_notion: NotionName (``"aleatoric"|"epistemic"|"total"``) resolved
            against the decomposition via :func:`select_uncertainty` (falls back when missing).
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
        uncertainty_notion: NotionName = "epistemic",
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
            uncertainty_notion: NotionName resolved via :func:`select_uncertainty`.
        """
        super().__init__(
            cfg=cfg,
            base_model_name=base_model_name,
            method_name=method_name,
            method_params=method_params,
            num_classes=num_classes,
            device=device,
            train_kwargs=train_kwargs,
            in_features=in_features,
        )
        self.rep_kwargs = rep_kwargs or {}
        self.uncertainty_notion = uncertainty_notion

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via the ``decide`` dispatch."""
        self.model.eval()
        amp_enabled = self.cfg.get("amp", False)
        parts = []
        for i in range(0, len(x), self.cfg.batch_size):
            xb = x[i : i + self.cfg.batch_size].float().to(self.device)
            with torch.amp.autocast(self.device.type, enabled=amp_enabled):
                probs = decide(self.model, xb, rep_kwargs=self.rep_kwargs).probabilities
            parts.append(probs.float().cpu())  # ty: ignore[unresolved-attribute]
        return torch.cat(parts)

    @torch.no_grad()
    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample uncertainty via representer -> quantify -> select_uncertainty."""
        self.model.eval()
        rep = representer(self.model, **self.rep_kwargs)
        amp_enabled = self.cfg.get("amp", False)
        parts: list[torch.Tensor] = []
        for i in range(0, len(x), self.cfg.batch_size):
            xb = x[i : i + self.cfg.batch_size].float().to(self.device)
            with torch.amp.autocast(self.device.type, enabled=amp_enabled):
                scores = cast("torch.Tensor", select_uncertainty(quantify(rep.predict(xb)), self.uncertainty_notion))
            parts.append(scores.detach().float().cpu())
        return torch.cat(parts)
