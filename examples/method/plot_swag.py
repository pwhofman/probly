"""=================
SWAG on Two Moons
=================

SWAG (SWA-Gaussian, :cite:`maddoxSimpleBaseline2019`) fits a Gaussian
distribution to the weights visited by SGD: its mean is the stochastic
weight average (SWA) and its covariance combines a diagonal term with a
low-rank term formed from the last snapshots of the SGD trajectory.
Sampling weight vectors from this Gaussian at inference time yields a
distribution over predictions from a single training run.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.swag import collect_swag, swag
from probly.representer import representer

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----
#
# ``swag`` wraps a copy of the base model; the wrapper is trained like the
# base model itself.

base_model = MLPClassifier()

swag_model = swag(
    base_model,
    max_rank=20,  # number of columns of the low-rank deviation matrix
    scale=0.5,  # the paper's 1/2 covariance scaling
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Standard SGD training. During the final epochs, ``collect_swag`` records a
# weight snapshot once per epoch, updating the running weight moments and the
# low-rank deviation matrix that define the SWAG posterior.

opt = torch.optim.SGD(swag_model.parameters(), lr=0.05, momentum=0.9)

swag_model.train()
for epoch in range(300):
    opt.zero_grad()
    out = swag_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    loss.backward()
    opt.step()
    if epoch >= 200:
        collect_swag(swag_model)

# %%
# Uncertainty Evaluation
# ----------------------
#
# Every repeated prediction drawn by the representer samples a fresh weight
# vector from the SWAG posterior; the original weights are restored afterwards.

swag_model.eval()
rep = representer(swag_model, num_samples=100)

plot = plot_example_uncertainty(X, y, rep, title="SWAG Predictive Uncertainty", notion="total")
plot.show()
