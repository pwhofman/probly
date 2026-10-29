"""============================================
Ensemble Selective Prediction for Regression
============================================

Abstain where the members of a regression ensemble disagree, at a calibrated coverage.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.metrics.selective_prediction import risk_coverage_curve
from probly.representer import representer
from probly.selective_prediction import selective_predictor, selector
from probly.transformation import ensemble

BLUE, RED, GRAY = "#1e88e5", "#ff0d57", "#7f8c8d"
NOISE = 0.12

# %%
# Data
# ----
#
# The training inputs leave a gap around zero, so the model never sees data there.

rng = np.random.default_rng(0)
X_train = torch.from_numpy(np.concatenate([rng.uniform(-4.5, -1, 40), rng.uniform(1, 4.5, 40)])).float()[:, None]
y_train = torch.sin(1.5 * X_train) + NOISE * torch.from_numpy(rng.standard_normal(X_train.shape)).float()


def sine_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(-4.5, 4.5, n)[:, None]
    noise = torch.randn(X.shape, generator=torch.Generator().manual_seed(seed))
    return X, (torch.sin(1.5 * X) + NOISE * noise).squeeze(-1)


X_calib, _ = sine_data(300, seed=1)
X_test, y_test = sine_data(400, seed=2)

# %%
# Transform, then train
# ---------------------

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
model = ensemble(net, num_members=10, reset_params=True)

for member in model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        nn.functional.mse_loss(member(X_train), y_train).backward()
        opt.step()

# %%
# Selective prediction
# --------------------
#
# The default notion, decider and loss assume a classifier, so regression passes a function for each: the variance
# across members as the criterion, their mean as the decision, and the squared error as the loss. The coverage
# selector needs no labels to calibrate.

predictor = selective_predictor(
    representer(model),
    selector=selector(coverage=0.7, notion=lambda s: s.sample_var().squeeze(-1)),
    decider=lambda s: s.sample_mean().squeeze(-1),
    loss=lambda d, y: (d - y) ** 2,
)

with torch.no_grad():
    predictor.calibrate(None, X_calib)
    scores = predictor.evaluate(y_test, X_test)
    result = predictor.predict(X_test)
print({name: round(value, 4) for name, value in scores.items()})

# %%
# Predictions and risk-coverage curve
# -----------------------------------
#
# The members disagree in the gray gap without training data, so it is rejected first.

x = X_test.squeeze(-1).numpy()
mean, std = result.decision.numpy(), result.uncertainty.sqrt().numpy()
accepted = result.accepted.numpy()
coverage, risk, _ = risk_coverage_curve(result.uncertainty, (result.decision - y_test) ** 2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.axvspan(-1, 1, color=GRAY, alpha=0.1)
ax.plot(x, np.sin(1.5 * x), color=GRAY, lw=1, label="true function")
ax.fill_between(x, mean - 2 * std, mean + 2 * std, color=BLUE, alpha=0.15, label=r"mean $\pm$ 2 std")
ax.scatter(x[accepted], mean[accepted], s=8, color=BLUE, label="accepted")
ax.scatter(x[~accepted], mean[~accepted], s=8, color=RED, label="rejected")
ax.set(xlabel="$x$", ylabel="prediction", ylim=(-1.5, 2.0), title="Ensemble prediction")
ax.legend(loc="upper left", ncol=2)

ax = axes[1]
ax.plot(coverage.numpy(), risk.numpy(), color=BLUE)
ax.axhline(scores["risk@1"], color=GRAY, linestyle="--", label="no abstention")
ax.axhline(NOISE**2, color=GRAY, linestyle=":", label="noise floor")
ax.scatter(scores["selected:coverage"], scores["selected:risk"], color=RED, zorder=3, label="selector")
ax.set(xlabel="coverage", ylabel="selective MSE", title="Risk-coverage curve")
ax.legend()
plt.tight_layout()
plt.show()
