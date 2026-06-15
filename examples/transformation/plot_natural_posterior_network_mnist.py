"""========================================
Natural Posterior Network on MNIST
========================================

Natural Posterior Network (NatPN, :cite:`charpentierNaturalPosteriorNetwork2022`)
performs a Bayesian update against a single normalizing-flow density and a
configurable certainty budget; uncertainty grows with distance from the
training data, not just near decision boundaries.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.natural_posterior_network import natural_posterior_network
from probly.train.evidential.torch import postnet_loss
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
# Backbone Pre-training
# ---------------------
#
# Train the shared feature extractor with standard cross-entropy on all 60k
# samples.  The final classification layer is not needed after this phase.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
base_model.train()
for _epoch in range(10):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        out = base_model(X_flat)
        nn.functional.cross_entropy(out, y_batch).backward()
        opt.step()
        correct += (out.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Model
# -----
#
# Strip the final classification head so the normalizing flow receives 256D
# feature vectors.  Unlike Posterior Network, NatPN does not require per-class
# sample counts: a single ``alpha_prior`` plus a ``certainty_budget`` scales
# the flow log-density into Dirichlet evidence.

backbone = nn.Sequential(*list(base_model.net)[:-1])

natpn_model = natural_posterior_network(
    backbone,
    latent_dim=8,           # dimension of the normalizing-flow latent space
    num_classes=10,
    num_flows=6,            # number of flow steps; more = more expressive density model
    certainty_budget="normal",
    alpha_prior=1.0,
)

# %%
# Flow Training
# -------------
#
# Freeze the backbone and train only the latent encoder, BatchNorm, and the
# normalizing flow with the PostNet UCE loss.  Mean reduction keeps gradient
# magnitudes stable across mini-batches and matches the NatPN paper.

for p in backbone.parameters():
    p.requires_grad_(False)

trainable = [p for p in natpn_model.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=1e-3)

natpn_model.train()
for _epoch in range(10):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        alpha = natpn_model(X_flat)
        loss = postnet_loss(alpha, y_batch, entropy_weight=1e-5, reduction="mean")
        loss.backward()
        opt.step()
        correct += (alpha.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Predictions and Uncertainty Quantification
# ------------------------------------------
#
# ``quantify(rep.represent(X))`` on a Natural Posterior Network exposes a
# ``total`` slot defined as the Dirichlet's differential entropy (nats, often
# non-positive), which does not align with the bits-style scoring used by the
# other examples.  For OOD comparison we compute the predictive entropy of the
# mean Dirichlet ``alpha / sum(alpha)`` directly.

natpn_model.eval()

with torch.no_grad():
    alpha = natpn_model(X_test)
alpha_np = alpha.numpy()
mean_probs = alpha_np / alpha_np.sum(-1, keepdims=True)

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

eps = 1e-12
uncertainty = -(mean_probs * np.log(mean_probs + eps)).sum(-1) / np.log(2)

# %%
# Visualization
# -------------

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (Natural Posterior Network)",
)
plot.show()
