"""=================================================================
Heteroscedastic Variational Bayesian Last Layers on Two Moons
=================================================================

The heteroscedastic variant of Variational Bayesian Last Layers
(:cite:`harrisonVariationalBayesian2024`) augments the discriminative VBLL
classifier with an *input-dependent* logit noise.  A second weight posterior maps
the features to a per-class log-noise vector, so the noise variance
``sigma_k^2(x)`` varies across the input space, letting the model express more
aleatoric uncertainty where the labels are genuinely noisier instead of assuming a
single global noise level.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.vbll import find_vbll_layer, vbll
from probly.representer import representer
from probly.train.vbll import vbll_loss

from examples.utils.model import SequentialModel
from examples.utils.plotting import plot_example_uncertainty

torch.manual_seed(0)

# %%
# Setup
# -----
#
# Two Moons with input-dependent (heteroscedastic) Gaussian noise: points
# further from each class' core get a larger noise scale, mimicking
# spatially varying label/observation noise.

X, y = make_moons(n_samples=500, noise=0.0, random_state=0)
rng = np.random.default_rng(0)

noise_scale = np.zeros_like(X[:, 0])

x0 = X[y == 0, 0]
noise_scale[y == 0] = 0.05 + 0.4 * (np.max(x0) - x0) / (np.max(x0) - np.min(x0))

x1 = X[y == 1, 0]
noise_scale[y == 1] = 0.05 + 0.4 * (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))

X += rng.normal(scale=np.expand_dims(noise_scale, 1), size=X.shape)

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----
#
# ``vbll(..., variant="heteroscedastic")`` replaces the backbone's linear head
# with a ``HetVBLLLayer``: a Gaussian posterior over the logit weights plus a
# second posterior whose features-to-log-noise map gives an input-dependent
# noise variance.

vbll_model = vbll(SequentialModel(), variant="heteroscedastic", parameterization="dense")

# %%
# Training
# --------
#
# The heteroscedastic variant is trained with the reduced Knowles-Minka softmax
# bound from :cite:`harrisonVariationalBayesian2024`; ``vbll_loss`` dispatches to
# it based on the layer type.  As with the other VBLL layers, the loss needs the
# features feeding the layer, which we capture with a forward pre-hook.

vbll_layer = find_vbll_layer(vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))


def train_het_vbll(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    kl_weight = 1.0 / X.shape[0]
    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        model(X)  # populates captured_features via the pre-hook
        loss = vbll_loss(vbll_layer, captured_features["features"], y, kl_weight)
        loss.backward()
        opt.step()


train_het_vbll(vbll_model, X_tensor, y_tensor)

# %%
# Uncertainty Evaluation
# ----------------------
#
# The input-dependent noise head co-trains with the backbone and captures the
# spatially varying (aleatoric) label noise, highlighting the noisier outer arms
# of the moons where the injected noise scale is largest.

vbll_model.eval()
rep = representer(vbll_model, num_samples=800)

plot = plot_example_uncertainty(
    X,
    y,
    rep,
    title="Het-VBLL Aleatoric Uncertainty",
    notion="aleatoric",
)
plot.show()
