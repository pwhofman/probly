"""=================
DARE on MNIST
=================

DARE (Deep Anti-Regularized Ensembles) adds a per-member anti-regularization
term that activates once the task loss drops below a threshold. This pushes
each member's weights to larger magnitudes, preserving the diversity created
by different initializations and improving out-of-distribution detection.
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
from probly.method.dare import dare
from probly.train.dare.torch import dare_regularizer
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
# DARE wraps an ensemble of independent members. Each member is trained with
# an anti-regularization term that fires when the per-batch cross-entropy drops
# below `threshold`, pushing weights to larger norms and preserving diversity.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

dare_model = dare(
    base_model,
    num_members=3,
    reset_params=True,
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Train each member with cross-entropy minus the DARE anti-regularization term.
# The anti-regularizer only activates once the batch loss falls below threshold.

threshold = 0.3

dare_model.train()
for member in dare_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for _epoch in range(5):
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_flat = X_batch.view(-1, 28 * 28)
            opt.zero_grad()
            out = member(X_flat)
            loss = nn.functional.cross_entropy(out, y_batch)
            reg = dare_regularizer(member, device="cpu", loss=loss.detach(), threshold=threshold)
            (loss - reg).backward()
            opt.step()
            correct += (out.detach().argmax(-1) == y_batch).sum().item()
            total += len(y_batch)
        if correct / total >= 0.97:
            break

# %%
# Predictions and Uncertainty Quantification
# ------------------------------------------
#
# Collect per-member softmax probabilities and compute predictive entropy
# from the averaged distribution.

dare_model.eval()
with torch.no_grad():
    member_probs = torch.stack(
        [m(X_test).softmax(-1) for m in dare_model]
    ).numpy()  # (num_members, N, 10)
mean_probs = member_probs.mean(0)

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
    member_probs=member_probs,
    is_ensemble=True,
    title="Top-5 Most Uncertain Test Predictions (DARE)",
)
plot.show()

# %%
# OOD Detection
# -------------
#
# DARE's anti-regularization is specifically designed to improve OOD detection
# by preserving member diversity.  Fashion MNIST provides a natural test:
# same image format as MNIST, but clothing categories the ensemble never saw.

ood_dataset = datasets.FashionMNIST(
    "~/.cache/mnist", train=False, download=True, transform=transforms.ToTensor()
)
X_ood = torch.stack([ood_dataset[i][0].flatten() for i in range(len(ood_dataset))])

with torch.no_grad():
    member_probs_ood = torch.stack(
        [m(X_ood).softmax(-1) for m in dare_model]
    ).numpy()
mean_probs_ood = member_probs_ood.mean(0)

uncertainty_ood = -(mean_probs_ood * np.log(mean_probs_ood + eps)).sum(-1) / np.log(2)

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
