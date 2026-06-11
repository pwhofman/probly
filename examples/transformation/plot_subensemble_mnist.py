"""============================
Sub-Ensemble on MNIST
============================

Share a frozen pre-trained backbone across several independent classification heads.
Useful when an expensive backbone is already trained and only lightweight heads should be replicated to obtain uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.quantification import quantify
from probly.representer import representer
from probly.transformation import subensemble
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

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
base_model.train()
for _epoch in range(5):
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
# ``subensemble`` requires an nn.Sequential for head_layer slicing.

subensemble_model = subensemble(
    base_model.net,
    num_heads=3,
    reset_params=True,
    head_layer=4,  # split point: lower = more diversity, higher = more sharing
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Only head parameters have requires_grad=True; the frozen backbone is skipped by the optimizer.

subensemble_model.train()
for member in subensemble_model:
    trainable = [p for p in member.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-3)
    for _epoch in range(5):
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_flat = X_batch.view(-1, 28 * 28)
            opt.zero_grad()
            out = member(X_flat)
            nn.functional.cross_entropy(out, y_batch).backward()
            opt.step()
            correct += (out.detach().argmax(-1) == y_batch).sum().item()
            total += len(y_batch)
        if correct / total >= 0.97:
            break

# %%
# Uncertainty Quantification
# --------------------------

subensemble_model.eval()
rep = representer(subensemble_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_total = uq.total
uncertainty = (
    _total.detach().numpy() if isinstance(_total, torch.Tensor) else np.asarray(_total)
)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------
#
# Collect per-member softmax probabilities.

with torch.no_grad():
    member_probs = torch.stack(
        [m(X_test).softmax(-1) for m in subensemble_model]
    ).numpy()  # (num_heads, N, 10)
mean_probs = member_probs.mean(0)

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------
#
# Plot the five most uncertain test digits with per-head agreement.

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (Sub-Ensemble)",
)
plot.show()
