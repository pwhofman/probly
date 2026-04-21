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

import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from probly.calibrator import calibrate
from probly.metrics._common import average_set_size, empirical_coverage_classification
from probly.method.conformal import conformal_aps, conformal_lac, conformal_raps, conformal_saps
from probly.predictor import LogitClassifier
from probly.representer import representer

torch.manual_seed(42)

# %%
# Data preparation
# ----------------

X, y = load_iris(return_X_y=True)
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


model = SimpleNet(4, 3)

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

calibrated_model = calibrate(conformal_lac(model), 0.05, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
lac_cov = empirical_coverage_classification(output, y_test_t)
lac_size = average_set_size(output)
print(f"LAC  — coverage: {lac_cov:.3f}, avg set size: {lac_size:.3f}")

# %%
# APS score
# ---------

calibrated_model = calibrate(conformal_aps(model), 0.05, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
aps_cov = empirical_coverage_classification(output, y_test_t)
aps_size = average_set_size(output)
print(f"APS  — coverage: {aps_cov:.3f}, avg set size: {aps_size:.3f}")

# %%
# RAPS score
# ----------

calibrated_model = calibrate(conformal_raps(model), 0.05, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
raps_cov = empirical_coverage_classification(output, y_test_t)
raps_size = average_set_size(output)
print(f"RAPS — coverage: {raps_cov:.3f}, avg set size: {raps_size:.3f}")

# %%
# SAPS score
# ----------

calibrated_model = calibrate(conformal_saps(model), 0.05, y_calib_t, X_calib_t)
output = representer(calibrated_model).predict(X_test_t)
saps_cov = empirical_coverage_classification(output, y_test_t)
saps_size = average_set_size(output)
print(f"SAPS — coverage: {saps_cov:.3f}, avg set size: {saps_size:.3f}")

# %%
# Summary
# -------

print("\n{:<6} {:>10} {:>14}".format("Score", "Coverage", "Avg set size"))
print("-" * 32)
for name, cov, sz in [
    ("LAC", lac_cov, lac_size),
    ("APS", aps_cov, aps_size),
    ("RAPS", raps_cov, raps_size),
    ("SAPS", saps_cov, saps_size),
]:
    print(f"{name:<6} {cov:>10.3f} {sz:>14.3f}")
