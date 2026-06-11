"""================================
Posterior Network on MNIST
================================

Output Dirichlet concentration parameters whose evidence comes from normalizing-flow density estimates in a learned latent space.
Uncertainty grows with distance from training data, not just near decision boundaries.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.transformation import posterior_network
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
# Strip the final classification head so the normalizing flows receive 256D
# feature vectors.  Pass actual per-class sample counts so the formula
# ``alpha = 1 + exp(log_density) * class_count`` produces a meaningful
# evidence scale.  With counts of 1 (the default) all alphas stay near 1
# and the Dirichlet is essentially uniform for every test input.

backbone = nn.Sequential(*list(base_model.net)[:-1])

all_labels = torch.cat([y for _, y in train_loader])
class_counts = torch.bincount(all_labels, minlength=10).tolist()

posterior_network_model = posterior_network(
    backbone,
    latent_dim=8,    # dimension of the normalizing-flow latent space
    num_classes=10,
    num_flows=6,     # number of flow steps per class; more = more expressive density model
    class_counts=class_counts,
    predictor_type="logit_classifier",
)

# %%
# Flow Training
# -------------
#
# Freeze the backbone and train only the latent encoder, BatchNorm, and
# normalizing flows with the PostNet UCE loss.  Mean reduction keeps
# gradient magnitudes stable across mini-batches.

for p in backbone.parameters():
    p.requires_grad_(False)

trainable = [p for p in posterior_network_model.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=1e-3)

posterior_network_model.train()
for _epoch in range(10):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        alpha = posterior_network_model(X_flat)
        loss = postnet_loss(alpha, y_batch, entropy_weight=1e-5)
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
# ``quantify(rep.represent(X))`` on a Posterior Network yields aleatoric and
# epistemic slots but no canonical ``total``.
# For OOD comparison with other methods we compute the predictive entropy
# of the mean Dirichlet ``alpha / sum(alpha)`` directly.

posterior_network_model.eval()

with torch.no_grad():
    alpha = posterior_network_model(X_test)
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
    title="Top-5 Most Uncertain Test Predictions (Posterior Network)",
)
plot.show()
