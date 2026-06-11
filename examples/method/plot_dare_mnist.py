"""=================
DARE on MNIST
=================

DARE (Deep Anti-Regularized Ensembles) adds a per-member anti-regularization
term that activates once the task loss drops below a threshold. This pushes
each member's weights to larger magnitudes, preserving the diversity created
by different initializations and improving out-of-distribution detection.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

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
    title="Top-5 Most Uncertain Test Predictions (DARE)",
)
plot.show()
