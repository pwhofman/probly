"""============================
Mahalanobis OOD on Two Moons
============================

The Mahalanobis out-of-distribution detector fits class-conditional Gaussians
with a *shared* covariance on a feature extractor, then scores each input by its
Mahalanobis distance to the nearest class centroid.  A single deterministic
forward pass yields both a class prediction and an epistemic / out-of-distribution
score.  This example also calibrates the multi-layer combination weights on a
small batch of synthetic out-of-distribution points.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

import torch
from torch import nn

from probly.representer import representer
from probly.method.mahalanobis import mahalanobis

from examples.utils.model import ResFFN
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# %%
# Model
# -----
#
# The transformation strips the classification head to expose penultimate
# features; the original head is kept for predictions.  No spectral
# normalization is applied -- the detector works on the plain feature extractor.

base_model = ResFFN()

mahalanobis_model = mahalanobis(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------

opt = torch.optim.Adam(mahalanobis_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

mahalanobis_model.train()
for epoch in range(200):
    opt.zero_grad()

    features = mahalanobis_model.encoder(X_tensor)
    logits = mahalanobis_model.classification_head(features)

    loss = criterion(logits, y_tensor)

    loss.backward()
    opt.step()

# %%
# Fit the Mahalanobis Heads
# -------------------------
#
# Estimate the per-class means and the shared covariance from the training
# features.  This only needs to happen once after training.

mahalanobis_model.eval()
mahalanobis_model.fit_mahalanobis_heads(X_tensor, y_tensor)

# %%
# Calibrate the Combiner
# ----------------------
#
# Fit the logistic-regression combination weights on the in-distribution data
# and a small batch of synthetic out-of-distribution points sampled far from the
# two moons.

rng = torch.Generator().manual_seed(0)
ood_tensor = torch.rand(500, 2, generator=rng) * 8.0 - 4.0

mahalanobis_model.fit_combiner(X_tensor, ood_tensor)

# %%
# Uncertainty Evaluation
# ----------------------

rep = representer(mahalanobis_model)

plot = plot_example_uncertainty(
    X, y, rep, title="Mahalanobis Out-of-Distribution Score", notion="epistemic", log_scale=True
)

plot.show()
