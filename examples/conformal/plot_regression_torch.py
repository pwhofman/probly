"""==========================================
Regression Conformal Prediction — PyTorch
==========================================

Demonstrate :func:`~probly.conformal.scores.absolute_error_score`
using a small :class:`~torch.nn.Module` on the Diabetes dataset.

After applying :func:`~probly.method.conformal.conformal_absolute_error`
and calibrating, :func:`~probly.representer.representer` expands the scalar prediction into
``[pred - q, pred + q]``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from probly.calibrator import calibrate
from probly.metrics._common import average_interval_size, empirical_coverage_regression
from probly.method.conformal import conformal_absolute_error
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
y_test_np = y_test  # keep numpy for plotting

# %%
# Define and train the model
# --------------------------


class SimpleNet(nn.Module):
    """Small MLP regressor."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


model = SimpleNet(X_train.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

model.train()
for _ in range(300):
    optimizer.zero_grad()
    loss_fn(model(X_train_t), y_train_t).backward()
    optimizer.step()
model.eval()

# %%
# Absolute error score
# --------------------

with torch.no_grad():
    calibrated_model = calibrate(conformal_absolute_error(model), 0.05, y_calib_t, X_calib_t)
    output = representer(calibrated_model).predict(X_test_t)

coverage = empirical_coverage_regression(output, y_test_t)
avg_size = average_interval_size(output)
print(f"Absolute Error — coverage: {coverage:.3f}, avg interval size: {avg_size:.1f}")

# %%
# Visualise prediction intervals
# -------------------------------

intervals = output.tensor.cpu().numpy()  # (n_test, 2)
order = np.argsort(y_test_np)

plt.figure(figsize=(9, 4))
plt.fill_between(
    range(len(y_test_np)),
    intervals[order, 0],
    intervals[order, 1],
    alpha=0.35,
    label="90% conformal interval",
)
plt.scatter(range(len(y_test_np)), y_test_np[order], s=15, color="tab:red", label="True value", zorder=3)
plt.xlabel("Test sample (sorted by true label)")
plt.ylabel("Target")
plt.title("Conformal Regression Intervals — Absolute Error Score (PyTorch)")
plt.legend()
plt.tight_layout()
plt.show()
