"""=====================================
Evidential Regression on a 1D Example
=====================================

Deep evidential regression obtains uncertainty estimates from a single
deterministic network. Roughly speaking, instead of predicting a Gaussian
directly, the network predicts a distribution over the parameters of that
Gaussian. More specifically, the last linear layer of a regressor is replaced
by a head that outputs the parameters ``(gamma, nu, alpha, beta)`` of a
Normal-Inverse-Gamma (NIG) distribution, which serves as a prior over the mean
``mu`` and variance ``sigma^2`` of a Gaussian likelihood. A single forward
pass therefore yields both a prediction and its uncertainty.

Following Amini et al. (2020), the NIG parameters give

* the prediction ``E[mu] = gamma``,
* the aleatoric uncertainty ``E[sigma^2] = beta / (alpha - 1)``,
* the epistemic uncertainty ``Var[mu] = beta / (nu * (alpha - 1))``.

As in the original paper, the model is trained on a noisy cubic function and
evaluated on a wider interval, which shows how the uncertainty estimates
behave outside the training data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from probly.losses.torch import evidential_nignll_loss, evidential_regression_regularization
from probly.method.evidential import evidential_regression

# %%
# Setup
# -----
#
# The targets are ``y = x^3 + eps`` with ``eps ~ N(0, 3^2)``. The training
# inputs cover ``[-4, 4]``; the evaluation grid spans ``[-6, 6]``.

x_train_np = np.random.uniform(-4, 4, (1000, 1)).astype(np.float32)
y_train_np = (x_train_np**3 + np.random.normal(0, 3, x_train_np.shape)).astype(np.float32)

X_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)
X_grid = torch.linspace(-6, 6, 500).reshape(-1, 1)

# %%
# Model
# -----
#
# The base model is an ordinary MLP regressor. ``evidential_regression``
# replaces its final ``nn.Linear`` with a Normal-Inverse-Gamma layer, whose
# output is a dict with the keys ``"gamma"``, ``"nu"``, ``"alpha"``, and
# ``"beta"``.


class MLPRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = evidential_regression(MLPRegressor())

# %%
# Training
# --------
#
# The loss is the NIG negative log-likelihood plus a regularizer that
# penalizes evidence on points with large errors. The regularizer thus
# discourages the model from being confident where its prediction is wrong.
# Training proceeds on random minibatches of 128 points.

opt = torch.optim.Adam(model.parameters(), lr=5e-3)
lam = 0.01

model.train()
for _step in range(2000):
    idx = torch.randint(0, len(X_train), (128,))
    opt.zero_grad()
    out = model(X_train[idx])
    loss = evidential_nignll_loss(out, y_train[idx]) + lam * evidential_regression_regularization(out, y_train[idx])
    loss.backward()
    opt.step()

# %%
# Uncertainty Estimates
# ---------------------
#
# The aleatoric and epistemic variances follow from the NIG parameters on the
# evaluation grid via the formulas above.

model.eval()
with torch.no_grad():
    out = model(X_grid)

gamma = out["gamma"].squeeze(-1).numpy()
nu = out["nu"].squeeze(-1).numpy()
alpha = out["alpha"].squeeze(-1).numpy()
beta = out["beta"].squeeze(-1).numpy()

aleatoric = beta / (alpha - 1)
epistemic = beta / (nu * (alpha - 1))
total_std = np.sqrt(aleatoric + epistemic)

# %%
# Visualization
# -------------
#
# The top panel shows the prediction together with a band of two standard
# deviations of the total uncertainty; the bottom panel separates the
# aleatoric from the epistemic variance. Outside the training range, where the
# model has seen no evidence, the epistemic variance grows.
#
# This decomposition should, however, be read with care. The predictive
# distribution of the NIG model is a Student-t distribution whose variance is
# the sum of the two terms, and the likelihood does not constrain how this
# sum is split between them. The network may therefore shift variance from
# one term to the other. In this example, the aleatoric part stays well below
# the true noise variance of ``9``, and the growth of the epistemic part
# outside the data depends on the initialization rather than being guaranteed
# by the method.

x_plot = X_grid.squeeze(-1).numpy()

fig, (ax_pred, ax_unc) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for ax in (ax_pred, ax_unc):
    ax.axvspan(-6, -4, color="gray", alpha=0.1)
    ax.axvspan(4, 6, color="gray", alpha=0.1)

ax_pred.scatter(x_train_np.ravel(), y_train_np.ravel(), s=10, color="gray", alpha=0.6, label="Train data")
ax_pred.plot(x_plot, x_plot**3, color="black", linestyle="--", linewidth=1, label="x^3")
ax_pred.plot(x_plot, gamma, color="tab:blue", linewidth=2, label="Prediction")
ax_pred.fill_between(
    x_plot,
    gamma - 2 * total_std,
    gamma + 2 * total_std,
    color="tab:blue",
    alpha=0.2,
    label="Total uncertainty (2 std)",
)
ax_pred.set_ylim(-150, 150)
ax_pred.set_ylabel("y")
ax_pred.legend(loc="upper left")
ax_pred.set_title("Evidential Regression Prediction")

ax_unc.plot(x_plot, epistemic, color="tab:green", linewidth=2, label="Epistemic variance")
ax_unc.plot(x_plot, aleatoric, color="tab:orange", linewidth=2, linestyle=":", label="Aleatoric variance")
ax_unc.set_yscale("log")
ax_unc.set_xlabel("x")
ax_unc.set_ylabel("Variance")
ax_unc.legend(loc="upper left")
ax_unc.set_title("Variance Decomposition")

fig.tight_layout()
plt.show()
