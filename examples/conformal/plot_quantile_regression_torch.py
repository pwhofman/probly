"""==================================================
Quantile Regression Conformal Prediction — PyTorch
==================================================

Demonstrate :func:`~probly.conformal.scores.cqr_score`,
:func:`~probly.conformal.scores.cqr_r_score`, and
:func:`~probly.conformal.scores.uacqr_score` using
PyTorch models on the Diabetes dataset.

**CQR / CQRr** use the standard conformal API
(:func:`~probly.calibrator.calibrate` + :func:`~probly.representer.representer`)
on a ``QuantileNet`` that outputs ``(n_samples, 2)`` lower/upper quantiles.

**UACQR** (Uncertainty-Aware CQR) normalises residuals by the standard
deviation of an ensemble of quantile predictors.  It expects predictions of
shape ``(n_members, n_samples, 2)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

from probly.calibrator import calibrate
from probly.metrics._common import average_interval_size, empirical_coverage_regression
from probly.method.conformal import conformal_cqr, conformal_cqr_r, conformal_uacqr
from probly.method.ensemble import EnsemblePredictor, ensemble
from probly.representer import representer

torch.manual_seed(42)

# %%
# Data preparation
# ----------------
ALPHA = 0.05
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

model = QuantileNet(X_train.shape[1])

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
    calibrated_model = calibrate(conformal_cqr(model), ALPHA, y_calib_t, X_calib_t)
    output = representer(calibrated_model).predict(X_test_t)

cqr_cov = empirical_coverage_regression(output, y_test_t)
cqr_size = average_interval_size(output)
print(f"CQR  — coverage: {cqr_cov:.3f}, avg interval size: {cqr_size:.1f}")

# %%
# CQRr score
# ----------

with torch.no_grad():
    calibrated_model = calibrate(conformal_cqr_r(model), ALPHA, y_calib_t, X_calib_t)
    output = representer(calibrated_model).predict(X_test_t)

cqrr_cov = empirical_coverage_regression(output, y_test_t)
cqrr_size = average_interval_size(output)
print(f"CQRr — coverage: {cqrr_cov:.3f}, avg interval size: {cqrr_size:.1f}")

# %%
# UACQR with an ensemble
# ----------------------
# ``uacqr_score`` normalises CQR residuals by the *standard deviation* across
# an ensemble of quantile predictors, making the conformal correction adaptive
# to regions of high model uncertainty.

ensemble_net = ensemble(QuantileNet(X_train.shape[1]), num_members=5)
member_optimizers = [torch.optim.Adam(member.parameters(), lr=0.01) for member in ensemble_net]


for _ in range(300):
    for m, member_optimizer in zip(ensemble_net, member_optimizers, strict=False):
        member_optimizer.zero_grad()
        quantile_loss(m(X_train_t), y_train_t, QUANTILES).backward()
        member_optimizer.step()


calibrated_model = calibrate(conformal_uacqr(ensemble_net), ALPHA, y_calib_t, X_calib_t)
representation = representer(calibrated_model)
output = representation.predict(X_test_t)

uacqr_cov = empirical_coverage_regression(output, y_test_t)
uacqr_size = average_interval_size(output)
print(f"UACQR — coverage: {uacqr_cov:.3f}, avg interval size: {uacqr_size:.1f}")

calibrated_model = calibrate(conformal_cqr(ensemble_net), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
cqr_cov_ens = empirical_coverage_regression(output, y_test_t)
cqr_size_ens = average_interval_size(output)
print(f"CQR (ensemble)  — coverage: {cqr_cov_ens:.3f}, avg interval size: {cqr_size_ens:.1f}")

calibrated_model = calibrate(conformal_cqr_r(ensemble_net), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
cqrr_cov_ens = empirical_coverage_regression(output, y_test_t)
cqrr_size_ens = average_interval_size(output)
print(f"CQRr (ensemble) — coverage: {cqrr_cov_ens:.3f}, avg interval size: {cqrr_size_ens:.1f}")


# %%
# Summary (Averaged over multiple runs)
# --------------------------------------
res = {"CQR": [], "CQRr": [], "UACQR": [], "CQR (ens)": [], "CQRr (ens)": []}
for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X)):
    torch.manual_seed(fold)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=fold)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_calib_t = torch.tensor(X_calib, dtype=torch.float32)
    y_calib_t = torch.tensor(y_calib, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Single QuantileNet
    fold_model = QuantileNet(X_train_t.shape[1])
    fold_opt = torch.optim.Adam(fold_model.parameters(), lr=0.01)
    fold_model.train()
    for _ in range(300):
        fold_opt.zero_grad()
        quantile_loss(fold_model(X_train_t), y_train_t, QUANTILES).backward()
        fold_opt.step()
    fold_model.eval()

    with torch.no_grad():
        for name, conformal_func in [("CQR", conformal_cqr), ("CQRr", conformal_cqr_r)]:
            calibrated_model = calibrate(conformal_func(fold_model), ALPHA, y_calib_t, X_calib_t)
            output = representer(calibrated_model).predict(X_test_t)
            cov = empirical_coverage_regression(output, y_test_t)
            size = average_interval_size(output)
            res[name].append((cov, size))

    # Ensemble QuantileNet
    ens_net = ensemble(QuantileNet(X_train_t.shape[1]), num_members=5)
    for m in ens_net:
        m.train()
        m_opt = torch.optim.Adam(m.parameters(), lr=0.01)
        for _ in range(300):
            m_opt.zero_grad()
            quantile_loss(m(X_train_t), y_train_t, QUANTILES).backward()
            m_opt.step()
        m.eval()

    for name, conformal_func in [("UACQR", conformal_uacqr), ("CQR (ens)", conformal_cqr), ("CQRr (ens)", conformal_cqr_r)]:
        calibrated_model = calibrate(conformal_func(ens_net), ALPHA, y_calib_t, X_calib_t)
        output = representer(calibrated_model).predict(X_test_t)
        cov = empirical_coverage_regression(output, y_test_t)
        size = average_interval_size(output)
        res[name].append((cov, size))

for name, vals in res.items():
    covs, sizes = zip(*vals)
    print(f"{name} — coverage: {np.mean(covs):.3f} ± {np.std(covs):.3f}, avg interval size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f}")
