"""PyTorch Dirichlet calibration wrapper."""

from __future__ import annotations

from typing import Self

import torch
from torch import nn
import torch.nn.functional as F

from probly.predictor import LogitClassifier, predict_raw

from ._common import DirichletCalibrationPredictor, dirichlet_calibration_generator

_LBFGS_MAX_ITER = 128


@DirichletCalibrationPredictor.register
class TorchDirichletCalibrationPredictor[**In](nn.Module):
    """Torch wrapper applying a fitted Dirichlet calibration map to logits."""

    predictor: nn.Module

    def __init__(self, predictor: nn.Module, num_classes: int, reg_lambda: float, reg_mu: float) -> None:
        """Initialise NaN-filled Dirichlet calibration buffers for the given class count."""
        super().__init__()
        self.predictor = predictor
        self.num_classes = int(num_classes)
        self.reg_lambda = float(reg_lambda)
        self.reg_mu = float(reg_mu)
        self.register_buffer("_dirichlet_weight", torch.full((num_classes, num_classes), float("nan")))
        self.register_buffer("_dirichlet_bias", torch.full((num_classes,), float("nan")))
        self.register_buffer("_is_calibrated", torch.tensor(False, dtype=torch.bool))

    @property
    def is_calibrated(self) -> bool:
        """Return whether the calibration map was fitted."""
        value = self._buffers.get("_is_calibrated")
        return isinstance(value, torch.Tensor) and bool(value.item())

    @property
    def weight(self) -> torch.Tensor | None:
        """Return the fitted weight matrix ``W`` when available."""
        if not self.is_calibrated:
            return None
        weight = self._buffers.get("_dirichlet_weight")
        return weight.detach().clone() if isinstance(weight, torch.Tensor) else None

    @property
    def bias(self) -> torch.Tensor | None:
        """Return the fitted bias ``b`` when available."""
        if not self.is_calibrated:
            return None
        bias = self._buffers.get("_dirichlet_bias")
        return bias.detach().clone() if isinstance(bias, torch.Tensor) else None

    def _require_calibrated(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Call calibrate() before prediction."
            raise ValueError(msg)
        weight = self._buffers.get("_dirichlet_weight")
        bias = self._buffers.get("_dirichlet_bias")
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            msg = "Dirichlet calibration buffers are missing."
            raise TypeError(msg)
        return weight, bias

    def _log_probabilities(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        raw_logits = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_logits, torch.Tensor):
            msg = f"Torch Dirichlet calibration expects torch logits, got {type(raw_logits)}"
            raise TypeError(msg)
        if raw_logits.ndim < 2 or raw_logits.shape[-1] != self.num_classes:
            msg = (
                "Dirichlet calibration expects logits with an explicit class axis of size "
                f"{self.num_classes}, got shape {tuple(raw_logits.shape)}."
            )
            raise ValueError(msg)
        return F.log_softmax(raw_logits, dim=-1)

    def calibrate(self, y_calib: torch.Tensor, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the Dirichlet map on calibration data with ODIR regularisation."""
        log_p = self._log_probabilities(*calib_args, **calib_kwargs).detach()
        labels = y_calib if isinstance(y_calib, torch.Tensor) else torch.as_tensor(y_calib, device=log_p.device)
        labels = labels.to(device=log_p.device).reshape(-1)

        flat_log_p = log_p.reshape(-1, self.num_classes)
        if labels.numel() != flat_log_p.shape[0]:
            msg = (
                "Dirichlet calibration labels must match logits batch size. "
                f"Got {labels.numel()} labels for {flat_log_p.shape[0]} logits."
            )
            raise ValueError(msg)

        weight = nn.Parameter(torch.eye(self.num_classes, device=log_p.device, dtype=log_p.dtype))
        bias = nn.Parameter(torch.zeros(self.num_classes, device=log_p.device, dtype=log_p.dtype))
        optimizer = torch.optim.LBFGS([weight, bias], max_iter=_LBFGS_MAX_ITER, line_search_fn="strong_wolfe")

        labels_long = labels.long()
        off_diag_mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=log_p.device)
        off_diag_count = max(self.num_classes * (self.num_classes - 1), 1)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            calibrated = flat_log_p @ weight.T + bias
            loss = F.cross_entropy(calibrated, labels_long)
            loss = loss + self.reg_lambda * (weight[off_diag_mask] ** 2).sum() / off_diag_count
            loss = loss + self.reg_mu * (bias**2).sum() / self.num_classes
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self._dirichlet_weight = weight.detach().clone()
            self._dirichlet_bias = bias.detach().clone()
            flag = self._buffers.get("_is_calibrated")
            if isinstance(flag, torch.Tensor):
                flag.fill_(True)
        return self

    def fit(self, x_calib: torch.Tensor, y_calib: torch.Tensor) -> Self:
        """Fit alias mapping sklearn-style argument order to calibrate."""
        return self.calibrate(y_calib, x_calib)  # ty:ignore[invalid-argument-type]

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Apply the fitted Dirichlet calibration map and return calibrated logits."""
        weight, bias = self._require_calibrated()
        log_p = self._log_probabilities(*args, **kwargs)
        weight = weight.to(device=log_p.device, dtype=log_p.dtype)
        bias = bias.to(device=log_p.device, dtype=log_p.dtype)
        return log_p @ weight.T + bias


LogitClassifier.register(TorchDirichletCalibrationPredictor)


@dirichlet_calibration_generator.register(nn.Module)
def generate_torch_dirichlet_calibrator(
    base: nn.Module,
    num_classes: int,
    reg_lambda: float,
    reg_mu: float,
) -> TorchDirichletCalibrationPredictor:
    """Create a torch Dirichlet calibration wrapper."""
    return TorchDirichletCalibrationPredictor(base, num_classes, reg_lambda, reg_mu)
