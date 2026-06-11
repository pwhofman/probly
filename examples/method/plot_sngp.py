"""==================================
SNGP Distance Awareness on 2D Toys
==================================

Spectral-normalized Neural Gaussian Process (SNGP, :cite:`liuSimplePrincipled2020`) wraps a
deep backbone with spectral normalization and replaces its final dense layer
with a random Fourier feature approximation to a Gaussian Process.  Spectral
normalization preserves input-space distance in the feature map, so the GP
posterior variance grows smoothly as inputs move away from the training
distribution -- unlike a standard network that extrapolates confidently into
out-of-distribution regions.

This example replicates the toy experiment from Figure 1 of the SNGP paper on
the classic 2-D Two Moons and Blobs datasets using a 12-layer, 128-unit deep
residual network (ResFFN-12-128).
"""

from __future__ import annotations

from sklearn.datasets import make_blobs, make_moons
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.representer import representer

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import ResFFN

# %%
# Setup
# -----
#
# Prepare the Two Moons and Blobs datasets.

X_moons, y_moons = make_moons(n_samples=500, noise=0.05, random_state=0)
X_moons_tensor = torch.from_numpy(X_moons).float()
y_moons_tensor = torch.from_numpy(y_moons).long()


X_blobs, y_blobs = make_blobs(
    n_samples=500, centers=[[-1.0, -1.0], [1.0, 1.0]], cluster_std=0.5, random_state=0
)
X_blobs_tensor = torch.from_numpy(X_blobs).float()
y_blobs_tensor = torch.from_numpy(y_blobs).long()


# %%
# Model
# -----
#
# SNGP wraps the ResFFN backbone with spectral normalization and replaces its
# linear output head with a random Fourier feature approximation to a GP.

sngp_model_moons = sngp(
    ResFFN(),
    num_random_features=128,
    ridge_penalty=0.01,
    norm_multiplier=0.9,
    n_power_iterations=1,
)
sngp_model_blobs = sngp(
    ResFFN(),
    num_random_features=128,
    ridge_penalty=0.01,
    norm_multiplier=0.9,
    n_power_iterations=1,
)

# %%
# Training
# --------
#
# The GP precision matrix is reset at the start of every epoch so it
# accumulates statistics across the full training set.  The loss is
# cross-entropy on the GP MAP logits returned by the model.


def train_sngp(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 300) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _epoch in range(epochs):
        reset_precision_matrix(model)
        out = model(X)
        logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


train_sngp(sngp_model_moons, X_moons_tensor, y_moons_tensor)
train_sngp(sngp_model_blobs, X_blobs_tensor, y_blobs_tensor)

# %%
# Uncertainty Evaluation
# ----------------------
#
# Evaluate the predictive uncertainty over a 2-D grid using the representer.

sngp_model_moons.eval()
rep_moons = representer(sngp_model_moons, num_samples=800)

plot_moons = plot_example_uncertainty(
    X_moons,
    y_moons,
    rep_moons,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    title="SNGP Predictive Uncertainty",
    notion="total",
)
plot_moons.show()


sngp_model_blobs.eval()
rep_blobs = representer(sngp_model_blobs, num_samples=800)

plot_blobs = plot_example_uncertainty(
    X_blobs,
    y_blobs,
    rep_blobs,
    title="SNGP Predictive Uncertainty (Blobs)",
    xlim=(-5.0, 5.0),
    ylim=(-5.0, 5.0),
    notion="total",
)
plot_blobs.show()
