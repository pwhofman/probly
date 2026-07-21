"""========================
Student-t VBLL on MNIST
========================

The Student-t variant of Variational Bayesian Last Layers
(:cite:`harrisonVariationalBayesian2024`) also infers the logit-noise variance
through a Gamma posterior, so the Gaussian over logits becomes a heavier-tailed
Student-t and the noise scale is learned rather than fixed.
"""

from __future__ import annotations

import numpy as np
import torch

from probly.method.vbll import find_vbll_layer, vbll
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
# ``vbll(variant="student_t")`` replaces the classifier's linear head with a
# ``TVBLLLayer``: a Gaussian posterior over the weights plus a Gamma posterior over
# the per-class noise precision.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
vbll_model = vbll(base_model, variant="student_t", parameterization="dense")

# %%
# Training
# --------
#
# For the Student-t layer, ``vbll_loss`` dispatches to the reduced Knowles-Minka
# softmax bound.  It needs the features feeding the layer, which we capture with
# a forward pre-hook.

vbll_layer = find_vbll_layer(vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))

opt = torch.optim.Adam(vbll_model.parameters(), lr=1e-3)
kl_weight = 1.0 / len(train_loader.dataset)

vbll_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        mean, _var = vbll_model(X_flat)
        loss = vbll_loss(vbll_layer, captured_features["features"], y_batch, kl_weight)
        loss.backward()
        opt.step()
        correct += (mean.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------
#
# The representer samples logits from the (Student-t informed) predictive Gaussian
# and softmaxes them into a second-order distribution over the classes.

vbll_model.eval()
rep = representer(vbll_model, num_samples=800)

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
    mean, _var = vbll_model(X_test)
    mean_probs = mean.softmax(-1).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (Student-t VBLL)",
)
plot.show()
