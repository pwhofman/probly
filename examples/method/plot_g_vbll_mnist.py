"""===============
G-VBLL on MNIST
===============

The generative variant of Variational Bayesian Last Layers (G-VBLL,
:cite:`harrisonVariationalBayesian2024`) models a per-class Gaussian density in
feature space instead of a linear logit map.  Because each density decays
quadratically away from its mean, the predictive is distance-aware and reverts to
the uniform distribution far from the training data.
"""

from __future__ import annotations

import numpy as np
import torch

from probly.method.g_vbll import find_g_vbll_layer, g_vbll
from probly.quantification import quantify
from probly.representer import representer
from probly.train.vbll import vbll_loss
from probly_benchmark.data import load_mnist

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_mnist_uncertainty

# %%
# Setup
# -----

train_loader, test_loader = load_mnist(batch_size=256)

X_test_batches, y_test_batches = zip(*test_loader)
X_test = torch.cat([x.view(-1, 28 * 28) for x in X_test_batches])
y_test = torch.cat(list(y_test_batches))
images_test = (X_test.view(-1, 28, 28) * 255).byte()

# %%
# Model
# -----
#
# ``g_vbll`` replaces the classifier's linear head with a ``GVBLLLayer`` that
# models a per-class Gaussian density over the features.  Its forward returns the
# class-conditional log-densities, used directly as categorical logits.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
g_vbll_model = g_vbll(base_model)

# %%
# Training
# --------
#
# For the generative layer, ``vbll_loss`` dispatches to the generative ELBO -- the
# Jensen bound on the class-conditional log-likelihood plus the class-mean KL and
# noise Wishart terms.  It needs the features feeding the layer, which we capture
# with a forward pre-hook.

vbll_layer = find_g_vbll_layer(g_vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))

opt = torch.optim.Adam(g_vbll_model.parameters(), lr=1e-3)
kl_weight = 1.0 / len(train_loader.dataset)

g_vbll_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        logits = g_vbll_model(X_flat)
        loss = vbll_loss(vbll_layer, captured_features["features"], y_batch, kl_weight)
        loss.backward()
        opt.step()
        correct += (logits.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------
#
# The G-VBLL predictive is a deterministic categorical distribution, so its total
# uncertainty is the entropy of the softmaxed class densities.

g_vbll_model.eval()
rep = representer(g_vbll_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
uncertainty = uq.total.detach().numpy() / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    logits = g_vbll_model(X_test)
    mean_probs = logits.softmax(-1).numpy()

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (G-VBLL)",
)
plot.show()
