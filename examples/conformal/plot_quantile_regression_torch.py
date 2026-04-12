"""=================================================
Quantile Regression Conformal Prediction — PyTorch
=================================================

Demonstrate :class:`~probly.conformal_prediction.scores_new.CQRScore`,
:class:`~probly.conformal_prediction.scores_new.CQRrScore`, and
:class:`~probly.conformal_prediction.scores_new.UACQRScore` using
PyTorch models on the Diabetes dataset.

**CQR / CQRr** use the standard conformal API
(:func:`~probly.calibrator.calibrate` + :func:`~probly.representer.representer`)
on a ``QuantileNet`` that outputs ``(n_samples, 2)`` lower/upper quantiles.

**UACQR** (Uncertainty-Aware CQR) normalises residuals by the standard
deviation of an ensemble of quantile predictors.  It expects predictions of
shape ``(n_members, n_samples, 2)`` and uses a manual calibration loop because
the standard :class:`~probly.layers.torch.ConformalQuantileHead` assumes a
``(n_samples, 2)`` forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from probly.calibrator import calibrate
from probly.conformal.metrics import average_interval_size, empirical_coverage_regression
from probly.conformal.methods.quantile_regression import conformalize_quantile_regressor
from probly.conformal.quantile import calculate_quantile
from probly.conformal.scores import CQRScore, CQRrScore, UACQRScore
from probly.method.ensemble._common import ensemble
from probly.representer import representer

torch.manual_seed(42)

# %%
# Data preparation
# ----------------

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_calib_t = torch.tensor(X_calib, dtype=torch.float32)
y_calib_t = torch.tensor(y_calib, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
y_test_np = y_test  # keep numpy for UACQR manual path

# %%
# Quantile loss
# -------------


def quantile_loss(output: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """Pinball loss summed over quantiles."""
    quantiles_t = torch.tensor(quantiles, dtype=torch.float32).unsqueeze(0)
    errors = target.unsqueeze(1) - output
    losses = torch.max((quantiles_t - 1) * errors, quantiles_t * errors)
    return torch.mean(torch.sum(losses, dim=1))


# %%
# QuantileNet for CQR and CQRr
# ----------------------------


class QuantileNet(nn.Module):
    """MLP that jointly predicts lower and upper quantiles."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Enforce lower_q <= upper_q so width-sensitive scores (CQRr) behave correctly
        return torch.sort(out, dim=-1).values  # (n_samples, 2): [lower_q, upper_q]


QUANTILES = [0.05, 0.95]

model = conformalize_quantile_regressor(QuantileNet(X_train.shape[1]))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for _ in range(300):
    optimizer.zero_grad()
    quantile_loss(model(X_train_t), y_train_t, QUANTILES).backward()
    optimizer.step()
model.eval()

# %%
# CQR score
# ---------

with torch.no_grad():
    calibrate(model, X_calib_t, y_calib_t, CQRScore(), alpha=0.05)
    output = representer(model).predict(X_test_t)

cqr_cov = empirical_coverage_regression(output, y_test_t)
cqr_size = average_interval_size(output)
print(f"CQR  — coverage: {cqr_cov:.3f}, avg interval size: {cqr_size:.1f}")

# %%
# CQRr score
# ----------

with torch.no_grad():
    calibrate(model, X_calib_t, y_calib_t, CQRrScore(), alpha=0.05)
    output = representer(model).predict(X_test_t)

cqrr_cov = empirical_coverage_regression(output, y_test_t)
cqrr_size = average_interval_size(output)
print(f"CQRr — coverage: {cqrr_cov:.3f}, avg interval size: {cqrr_size:.1f}")

# %%
# UACQR with an ensemble
# ----------------------
# ``UACQRScore`` normalises CQR residuals by the *standard deviation* across
# an ensemble of quantile predictors, making the conformal correction adaptive
# to regions of high model uncertainty.
#
# The score expects predictions of shape ``(n_members, n_samples, 2)``.
# Because :class:`~probly.layers.torch.ConformalQuantileHead` assumes
# ``(n_samples, 2)`` in its calibrated forward pass, UACQR uses a manual
# calibration loop instead of the standard ``representer`` API.

ensemble_net = ensemble(QuantileNet(X_train.shape[1]), num_members=5)
model = conformalize_quantile_regressor(ensemble_net)


for _ in range(300):
    for m in model:
        optimizer.zero_grad()
        quantile_loss(m(X_train_t), y_train_t, QUANTILES).backward()
        optimizer.step()


calibrate(model, X_calib_t, y_calib_t, UACQRScore(), alpha=0.05)
representation = representer(model)
output = representation.predict(X_test_t)

uacqr_cov = empirical_coverage_regression(output, y_test_t)
uacqr_size = average_interval_size(output)
print(f"UACQR — coverage: {uacqr_cov:.3f}, avg interval size: {uacqr_size:.1f}")

# %%
# Comparison
# ----------

print("\n{:<6} {:>10} {:>18}".format("Score", "Coverage", "Avg interval size"))
print("-" * 36)
for name, cov, sz in [
    ("CQR", cqr_cov, cqr_size),
    ("CQRr", cqrr_cov, cqrr_size),
    ("UACQR", uacqr_cov, uacqr_size),
]:
    print(f"{name:<6} {cov:>10.3f} {sz:>18.1f}")
