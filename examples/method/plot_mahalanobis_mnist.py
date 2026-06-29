"""========================
Mahalanobis OOD on MNIST
========================

The Mahalanobis out-of-distribution detector fits class-conditional Gaussians
with a *shared* covariance on a feature extractor, then scores each input by its
Mahalanobis distance to the nearest class centroid.  A single deterministic
forward pass yields both a class prediction and an epistemic / out-of-distribution
score.  The multi-layer combination weights are calibrated on a batch of random
noise images that act as an out-of-distribution proxy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.mahalanobis import mahalanobis
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.data import load_mnist

from examples.utils.model import ResFFN
from examples.utils.plotting import plot_mnist_uncertainty

# %%
# Setup
# -----

train_loader, test_loader = load_mnist(batch_size=256)

X_train_batches, y_train_batches = zip(*train_loader)
X_train = torch.cat([x.view(-1, 28 * 28) for x in X_train_batches])
y_train = torch.cat(list(y_train_batches))

X_test_batches, y_test_batches = zip(*test_loader)
X_test = torch.cat([x.view(-1, 28 * 28) for x in X_test_batches])
y_test = torch.cat(list(y_test_batches))
images_test = (X_test.view(-1, 28, 28) * 255).byte()

# %%
# Model
# -----
#
# The transformation strips the classification head to expose penultimate
# features and keeps the original head for predictions.

base_model = ResFFN(in_features=28 * 28, hidden_features=256, out_features=10)

mahalanobis_model = mahalanobis(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------

opt = torch.optim.Adam(mahalanobis_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

mahalanobis_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        features = mahalanobis_model.encoder(X_flat)
        logits = mahalanobis_model.classification_head(features)

        loss = criterion(logits, y_batch)

        loss.backward()
        opt.step()
        correct += (logits.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Fit the Mahalanobis Heads
# -------------------------
#
# Estimate the per-class means and the shared covariance from the training
# features in a single pass.

mahalanobis_model.eval()
mahalanobis_model.fit_mahalanobis_heads(X_train, y_train)

# %%
# Calibrate the Combiner
# ----------------------
#
# Use random noise images as an out-of-distribution proxy to calibrate the
# logistic-regression combination weights against the in-distribution data.

rng = torch.Generator().manual_seed(0)
ood_noise = torch.rand(1000, 28 * 28, generator=rng)

mahalanobis_model.fit_combiner(X_train[:1000], ood_noise)

# %%
# Uncertainty Quantification
# --------------------------

rep = representer(mahalanobis_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_unc = uq.epistemic if hasattr(uq, "epistemic") else uq.total
uncertainty = _unc.detach().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    logits, _ = mahalanobis_model(X_test)
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
    title="Top-5 Most Uncertain Test Predictions (Mahalanobis)",
)
plot.show()
