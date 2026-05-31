"""============================
BatchEnsemble on MNIST
============================

Replace a full ensemble with per-member rank-1 multiplicative factors on top of a shared backbone.
Training is two-phase: the backbone is pre-trained first, then per-member factors are fine-tuned on a tiled batch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
from probly.quantification import quantify
from probly.representer import representer
from probly.transformation import batchensemble
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
# Train the shared backbone with standard cross-entropy before wrapping it
# as a BatchEnsemble so the shared weights start in a sensible region.

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
# Per-member rank-1 factor vectors ``r`` and ``s`` rescale the shared weight matrix:
# the effective weight for member ``i`` is ``diag(s_i) * W * diag(r_i)``.
# Initializing both near 1.0 keeps members close to the pre-trained backbone at the
# start; the std controls how much they diverge.

num_members = 3
batchensemble_model = batchensemble(
    base_model,
    num_members=num_members,
    use_base_weights=True,  # seed the shared backbone with the pre-trained weights
    r_mean=1.0,             # input-scale factor, identity at 1.0
    r_std=0.5,              # controls input-scale diversity across members
    s_mean=1.0,             # output-scale factor, identity at 1.0
    s_std=0.5,              # controls output-scale diversity across members
    predictor_type="logit_classifier",
)

# %%
# Fine-tuning
# -----------
#
# Fine-tune the rank-1 factors (and shared weights) with standard cross-entropy
# on a tiled batch.  The batch must be repeated ``num_members`` times so that
# member ``i`` processes samples ``[i*B : (i+1)*B]`` in a single forward pass.

batchensemble_model.train()
opt = torch.optim.Adam(batchensemble_model.parameters(), lr=1e-4)

for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_tiled = X_batch.view(-1, 28 * 28).repeat(num_members, 1)
        y_tiled = y_batch.repeat(num_members)
        opt.zero_grad()
        out = batchensemble_model(X_tiled)
        nn.functional.cross_entropy(out, y_tiled).backward()
        opt.step()
        correct += (out.detach().argmax(-1) == y_tiled).sum().item()
        total += len(y_tiled)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------

batchensemble_model.eval()
rep = representer(batchensemble_model, num_samples=300)

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
# Run all members in a single tiled forward pass and extract per-member
# probabilities.

N = len(X_test)
with torch.no_grad():
    X_tiled = X_test.repeat(num_members, 1)
    out = batchensemble_model(X_tiled)
    member_probs = out.softmax(-1).reshape(num_members, N, 10).numpy()
mean_probs = member_probs.mean(0)

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------
#
# Plot the five most uncertain test digits with per-member agreement.

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    member_probs=member_probs,
    is_ensemble=True,
    title="Top-5 Most Uncertain Test Predictions (BatchEnsemble)",
)
plot.show()

# %%
# OOD Detection
# -------------
#
# Fashion MNIST shares MNIST's image format (28x28 grayscale) but contains
# clothing categories the ensemble was never trained on.  The same representer
# from the UQ section is applied to the OOD data -- no new inference setup needed.

ood_dataset = datasets.FashionMNIST(
    "~/.cache/mnist", train=False, download=True, transform=transforms.ToTensor()
)
X_ood = torch.stack([ood_dataset[i][0].flatten() for i in range(len(ood_dataset))])

with torch.no_grad():
    representation_ood = rep.represent(X_ood)

uq_ood = quantify(representation_ood)
_total_ood = uq_ood.total
uncertainty_ood = (
    _total_ood.detach().numpy() if isinstance(_total_ood, torch.Tensor) else np.asarray(_total_ood)
)
uncertainty_ood = uncertainty_ood / np.log(2)
if uncertainty_ood.ndim > 1:
    uncertainty_ood = uncertainty_ood.sum(axis=-1)

# %%
# Score Histogram
# ---------------
#
# The ID distribution should be tightly concentrated near zero (high confidence);
# the OOD distribution should shift right (higher entropy).

fig = plot_histogram(
    uncertainty,
    uncertainty_ood,
    title="Predictive Entropy: MNIST (ID) vs Fashion MNIST (OOD)",
)
fig.axes[0].set_xlabel("Predictive Entropy (bits)")
plt.show()

# %%
# ROC Curve
# ---------
#
# AUROC measures how well entropy separates ID from OOD samples end-to-end.
# A perfect detector scores 1.0; random guessing scores 0.5.

ood_metrics = evaluate_ood(uncertainty, uncertainty_ood, metrics=["auroc", "fpr"])
labels = np.concatenate([np.zeros(len(uncertainty)), np.ones(len(uncertainty_ood))])
preds = np.concatenate([uncertainty, uncertainty_ood])
fpr_curve, tpr_curve, _ = roc_curve(labels, preds)

fig = plot_roc_curve(
    fpr_curve,
    tpr_curve,
    auroc=ood_metrics["auroc"],
    fpr95=ood_metrics["fpr"],
)
plt.show()
