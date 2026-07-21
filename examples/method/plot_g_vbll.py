"""========================================================
Generative Variational Bayesian Last Layers on Two Moons
========================================================

The generative variant of Variational Bayesian Last Layers (G-VBLL,
:cite:`harrisonVariationalBayesian2024`) replaces a network's final dense layer
with a *generative* classifier: each class owns a Gaussian density in feature
space, and the predictive class probabilities are the softmax of the
class-conditional log-densities.  Because each density decays quadratically away
from its class mean, the predictive is distance-aware -- it reverts to the uniform
distribution far from the training data, unlike the discriminative VBLL which can
extrapolate confidently out of distribution.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.g_vbll import find_g_vbll_layer, g_vbll
from probly.representer import representer
from probly.train.vbll import vbll_loss

from examples.utils.model import ResFFN
from examples.utils.plotting import plot_example_uncertainty

torch.manual_seed(0)

# %%
# Setup
# -----
#
# Prepare the Two Moons dataset.

X, y = make_moons(n_samples=500, noise=0.1, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----
#
# ``g_vbll`` replaces the ResFFN backbone's linear output head with a
# ``GVBLLLayer`` that models a per-class Gaussian density over the residual
# features.

g_vbll_model = g_vbll(ResFFN())

# %%
# Training
# --------
#
# G-VBLL is fit by maximizing the generative ELBO of
# :cite:`harrisonVariationalBayesian2024`; ``vbll_loss`` dispatches to it based on
# the layer type.  As with the other VBLL layers, the loss needs the features
# feeding the layer, which we capture with a forward pre-hook.

vbll_layer = find_g_vbll_layer(g_vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))


def train_g_vbll(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1500) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    kl_weight = 1.0 / X.shape[0]
    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        model(X)  # populates captured_features via the pre-hook
        loss = vbll_loss(vbll_layer, captured_features["features"], y, kl_weight)
        loss.backward()
        opt.step()


train_g_vbll(g_vbll_model, X_tensor, y_tensor)

# %%
# Predictive Uncertainty
# ----------------------
#
# The G-VBLL predictive is a deterministic categorical distribution, so its total
# uncertainty is the entropy of the softmaxed class densities -- low on the moons
# and growing in the data-free regions around them.

g_vbll_model.eval()
rep = representer(g_vbll_model)

plot = plot_example_uncertainty(
    X,
    y,
    rep,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    title="G-VBLL Predictive Uncertainty",
    notion="total",
)
plot.show()
