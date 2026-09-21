"""=============
SWAG on MNIST
=============

SWAG (SWA-Gaussian, :cite:`maddoxSimpleBaseline2019`) fits a Gaussian posterior over the network weights from
snapshots of the late SGD trajectory: the mean is the stochastic weight average (SWA) and the covariance combines
a diagonal term with a low-rank term. Sampling weight vectors from this posterior at inference time yields a
distribution over predictions from a single training run.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.swag import collect_swag, swag
from probly.quantification import quantify
from probly.representer import representer
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
# ``swag`` wraps a copy of the base model; the wrapper is trained exactly like the base model itself.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
flat_model = nn.Sequential(nn.Flatten(), base_model)

swag_model = swag(flat_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# Standard SGD training with a constant learning rate. After a warmup phase, ``collect_swag`` records one weight
# snapshot per epoch; these snapshots define the SWAG posterior.

opt = torch.optim.SGD(swag_model.parameters(), lr=0.05, momentum=0.9)

epochs, warmup_epochs = 10, 5

swag_model.train()
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(swag_model(X_batch), y_batch)
        loss.backward()
        opt.step()
    if epoch >= warmup_epochs:
        collect_swag(swag_model)

# %%
# Uncertainty Quantification
# --------------------------
#
# Every repeated prediction drawn by the representer samples a fresh weight vector from the SWAG posterior; the
# model's own weights are left untouched.

swag_model.eval()
rep = representer(swag_model, num_samples=30)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_total = uq.total
uncertainty = _total.detach().numpy() if isinstance(_total, torch.Tensor) else np.asarray(_total)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------
#
# Averaging the per-sample probabilities gives the Bayesian model average over the SWAG posterior. Loading the
# SWA mean weights into the model gives the deterministic SWA prediction for comparison.

mean_probs = representation.tensor.probabilities.mean(1).numpy()  # (N, num_samples, 10) -> (N, 10)
accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"SWAG (Bayesian model average) test accuracy: {accuracy:.1f}%")

swag_model.load_mean_parameters()
with torch.no_grad():
    swa_probs = swag_model(X_test).softmax(-1).numpy()
swa_accuracy = (swa_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"SWA (mean weights) test accuracy: {swa_accuracy:.1f}%")

# %%
# Visualization
# -------------

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (SWAG)",
)
plot.show()
