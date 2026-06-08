"""====================
Het-Net on Two Moons
====================

Het-Net augments a standard classifier with a learnable heteroscedastic noise
head that draws Monte Carlo samples in logit space.  The representer reuses the
same noise mechanism at inference time to estimate per-sample aleatoric
uncertainty, which is helpful on data with input-dependent label noise such as
this noisy Two Moons setup.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
import torch.nn.functional as F
import numpy as np

from probly.layers.torch import HeteroscedasticLayer
from probly.method.het_net import het_net
from probly.representer import representer

from examples.utils.model import SequentialModel
from examples.utils.plotting import plot_example_uncertainty

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
# Het-Net wraps the backbone with a ``HeteroscedasticLayer`` that learns a
# per-sample noise distribution over the logits.  The noise head is co-trained
# with the backbone and captures input-dependent (aleatoric) uncertainty.

base_model = SequentialModel()
het_net_model = het_net(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# Setting ``training_samples = S`` on every ``HeteroscedasticLayer`` makes the
# head draw S noise samples per input in a single vectorized forward pass and
# return the log of the softmax-averaged probabilities, optimized with NLL.

opt = torch.optim.Adam(het_net_model.parameters(), lr=1e-3)
training_samples = 4

het_layers = [m for m in het_net_model.modules() if isinstance(m, HeteroscedasticLayer)]
for layer in het_layers:
    layer.training_samples = training_samples

het_net_model.train()
try:
    for _epoch in range(500):
        opt.zero_grad()
        log_probs = het_net_model(X_tensor)
        loss = F.nll_loss(log_probs, y_tensor)
        loss.backward()
        opt.step()
finally:
    for layer in het_layers:
        layer.training_samples = 1

# %%
# Uncertainty Evaluation
# ----------------------

het_net_model.eval()
rep = representer(het_net_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="HET-Net Predictive Uncertainty", notion="aleatoric")
plot.show()
