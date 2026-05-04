"""=============================================
Classification Conformal Prediction — PyTorch
=============================================

Demonstrate all four classification non-conformity scores
(:func:`~probly.conformal.scores.lac_score`,
:class:`~probly.conformal.scores.APSScore`,
:class:`~probly.conformal.scores.RAPSScore`,
:class:`~probly.conformal.scores.SAPSScore`)
using a small :class:`~torch.nn.Module` on the Iris dataset.

Each score uses its own conformal wrapper. During calibration the conformal quantile
is computed; after calibration :func:`~probly.representer.representer` returns a
boolean inclusion mask (the conformal prediction set).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split

from probly.calibrator import calibrate
from probly.evaluation import coverage, efficiency
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps, conformal_saps
from probly.predictor import LogitClassifier
from probly.representer import representer

torch.manual_seed(42)

# %%
# Data preparation
# ----------------
ALPHA = 0.05
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_calib_t = torch.tensor(X_calib, dtype=torch.float32)
y_calib_t = torch.tensor(y_calib, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# %%
# Define and train the model
# --------------------------


class SimpleNet(nn.Module, LogitClassifier):
    """Two-layer classifier."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


model = SimpleNet(64, 10)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

model.train()
for _ in range(200):
    optimizer.zero_grad()
    loss_fn(model(X_train_t), y_train_t).backward()
    optimizer.step()
model.eval()

# %%
# LAC score
# ---------

calibrated_model = calibrate(conformal_lac(model), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
lac_cov = coverage(output, y_test_t)
lac_size = efficiency(output)
print(f"LAC  — coverage: {lac_cov:.3f}, avg set size: {lac_size:.3f}")

# %%
# APS score
# ---------

calibrated_model = calibrate(conformal_aps(model, randomized=True), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
aps_cov = coverage(output, y_test_t)
aps_size = efficiency(output)
print(f"APS  — coverage: {aps_cov:.3f}, avg set size: {aps_size:.3f}")

# %%
# RAPS score
# ----------

calibrated_model = calibrate(conformal_raps(model, randomized=True, lambda_reg=0.1, k_reg=0), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
raps_cov = coverage(output, y_test_t)
raps_size = efficiency(output)
print(f"RAPS — coverage: {raps_cov:.3f}, avg set size: {raps_size:.3f}")

# %%
# SAPS score
# ----------

calibrated_model = calibrate(conformal_saps(model, randomized=True, lambda_val=0.1), ALPHA, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
saps_cov = coverage(output, y_test_t)
saps_size = efficiency(output)
print(f"SAPS — coverage: {saps_cov:.3f}, avg set size: {saps_size:.3f}")

# %%
# Summary (Averaged over multiple runs)
# --------------------------------------
res = {
    "LAC": [],
    "APS": [],
    "RAPS": [],
    "SAPS": [],
}
for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(X)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=fold)

    fold_model = RandomForestClassifier(random_state=fold)
    fold_model.fit(X_train, y_train)
    for name, conformal_func in [("LAC", conformal_lac), ("APS", conformal_aps), ("RAPS", conformal_raps), ("SAPS", conformal_saps)]:
        calibrated_model = calibrate(conformal_func(fold_model), ALPHA, y_calib, X_calib)
        output = representer(calibrated_model).predict(X_test)
        cov = coverage(output, y_test)
        size = efficiency(output)
        res[name].append((cov, size))

for name, vals in res.items():
    covs, sizes = zip(*vals)
    print(f"{name} — coverage: {np.mean(covs):.3f} ± {np.std(covs):.3f}, avg set size: {np.mean(sizes):.3f} ± {np.std(sizes):.3f}")
