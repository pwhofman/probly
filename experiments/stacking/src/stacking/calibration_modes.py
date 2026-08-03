"""Shared plumbing for the four calibration modes used by the stack_* scripts.

Each composition script picks a training loss (cross-entropy or label
relaxation) and a post-hoc logit transform (none, temperature scaling,
or per-class isotonic) by name. This module only handles the
mode-dependent bits; nothing in here knows anything about the
underlying base method (DARE, credal-wrapper, HetNet).

Modes
-----

* ``"none"`` -- plain cross-entropy at train time, no post-hoc step.
* ``"temperature"`` -- cross-entropy at train time, ``temperature_scaling``
  fitted on calibration logits at the post-hoc step.
* ``"isotonic"`` -- cross-entropy at train time, one isotonic regressor
  per class fitted on calibration softmax probabilities (one-vs-rest),
  re-normalised after transform, then routed back to log-probability
  space so the rest of the pipeline can keep talking in logits. Probly's
  torch isotonic implementation is binary-only, so we use sklearn's
  ``IsotonicRegression`` directly.
* ``"label_relaxation"`` -- :class:`probly.train.calibration.torch.LabelRelaxationLoss`
  at train time (Lienen & Huellermeier 2021), no post-hoc step.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch
from torch import nn

from probly.calibrator import calibrate
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.predictor import predict_raw
from probly.train.calibration.torch import LabelRelaxationLoss

CALIBRATION_MODES: Final[tuple[str, ...]] = (
    "none",
    "temperature",
    "isotonic",
    "label_relaxation",
)
"""All supported calibration modes, in stable canonical order."""


def make_loss(mode: str, *, lr_alpha: float) -> nn.Module:
    """Return the training loss used for the chosen calibration mode.

    Args:
        mode: One of :data:`CALIBRATION_MODES`.
        lr_alpha: Alpha parameter for :class:`LabelRelaxationLoss`. Only
            consulted when ``mode == "label_relaxation"``.

    Returns:
        A loss module mapping ``(logits, targets) -> scalar``.

    Raises:
        ValueError: If ``mode`` is not a recognised mode.
    """
    if mode not in CALIBRATION_MODES:
        msg = f"unknown calibration mode: {mode!r}; expected one of {CALIBRATION_MODES}"
        raise ValueError(msg)
    if mode == "label_relaxation":
        return LabelRelaxationLoss(alpha=lr_alpha)
    return nn.CrossEntropyLoss()


def calibrate_logits(
    *,
    mode: str,
    calib_logits: torch.Tensor,
    test_logits: torch.Tensor,
    y_calib: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the post-hoc logit transform for the chosen mode.

    For ``"none"`` and ``"label_relaxation"`` the input logits are
    returned unchanged. For ``"temperature"`` and ``"isotonic"`` the
    returned tensors are calibrated counterparts of the inputs (still in
    logit space, so downstream conformal layers consume them
    unchanged).

    Args:
        mode: One of :data:`CALIBRATION_MODES`.
        calib_logits: Pooled logits on the calibration split, shape
            ``(n_calib, num_classes)``.
        test_logits: Pooled logits on the test split, shape
            ``(n_test, num_classes)``.
        y_calib: Hard labels on the calibration split, shape
            ``(n_calib,)``.
        num_classes: Number of classes.

    Returns:
        ``(calib_calibrated, test_calibrated)`` -- both float tensors on
        the same device and dtype as the inputs.

    Raises:
        ValueError: If ``mode`` is not a recognised mode.
    """
    if mode not in CALIBRATION_MODES:
        msg = f"unknown calibration mode: {mode!r}; expected one of {CALIBRATION_MODES}"
        raise ValueError(msg)
    if mode in {"none", "label_relaxation"}:
        return calib_logits, test_logits
    if mode == "temperature":
        return _temperature_transform(calib_logits=calib_logits, test_logits=test_logits, y_calib=y_calib)
    if mode == "isotonic":
        return _isotonic_transform(
            calib_logits=calib_logits,
            test_logits=test_logits,
            y_calib=y_calib,
            num_classes=num_classes,
        )
    msg = f"unhandled calibration mode: {mode!r}"
    raise RuntimeError(msg)


def _temperature_transform(
    *, calib_logits: torch.Tensor, test_logits: torch.Tensor, y_calib: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit temperature scaling on calib logits and apply to both splits."""
    cal = temperature_scaling(torch_identity_logit_model())
    calibrate(cal, y_calib, calib_logits)
    out_calib = predict_raw(cal, calib_logits)
    out_test = predict_raw(cal, test_logits)
    if not isinstance(out_calib, torch.Tensor) or not isinstance(out_test, torch.Tensor):
        msg = "temperature_scaling returned a non-tensor; expected torch logits"
        raise TypeError(msg)
    return out_calib, out_test


def _isotonic_transform(
    *,
    calib_logits: torch.Tensor,
    test_logits: torch.Tensor,
    y_calib: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class one-vs-rest isotonic on softmax, then back to log-probs.

    For each class ``k``, an ``IsotonicRegression`` from sklearn is
    fitted on the calibration softmax probability of ``k`` against the
    binary indicator ``y_calib == k``. At transform time the per-class
    regressors are applied independently and the resulting probability
    matrix is renormalised across classes. The renormalised
    probabilities are returned in log-space so downstream layers can
    keep operating on logits.
    """
    calib_probs = torch.softmax(calib_logits, dim=-1).detach().cpu().numpy()
    test_probs = torch.softmax(test_logits, dim=-1).detach().cpu().numpy()
    y_np = y_calib.detach().cpu().numpy().astype(np.int64)

    regressors: list[IsotonicRegression] = []
    for k in range(num_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(calib_probs[:, k], (y_np == k).astype(np.float64))
        regressors.append(iso)

    def _apply(probs: np.ndarray) -> np.ndarray:
        per_class = np.stack([regressors[k].transform(probs[:, k]) for k in range(num_classes)], axis=1)
        per_class = np.clip(per_class, 1e-12, 1.0)
        return per_class / per_class.sum(axis=1, keepdims=True)

    calib_calibrated = np.log(_apply(calib_probs))
    test_calibrated = np.log(_apply(test_probs))

    return (
        torch.as_tensor(calib_calibrated, dtype=calib_logits.dtype, device=calib_logits.device),
        torch.as_tensor(test_calibrated, dtype=test_logits.dtype, device=test_logits.device),
    )
