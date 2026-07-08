"""============================
MC Dropout on MNIST
============================

Keep dropout active at inference and average several stochastic forward passes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.quantification import quantify
from probly.representer import representer
from probly.transformation import dropout
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

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

dropout_model = dropout(
    base_model,
    p=0.1,  # zeroing probability (kept active at inference)
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Standard cross-entropy with mini-batches.  Dropout stays active at
# inference time, which is what enables repeated forward passes to produce
# a distribution over predictions.

opt = torch.optim.Adam(dropout_model.parameters(), lr=1e-3)

dropout_model.train()
for _epoch in range(10):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        out = dropout_model(X_flat)
        nn.functional.cross_entropy(out, y_batch).backward()
        opt.step()
        correct += (out.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------

dropout_model.eval()
rep = representer(dropout_model, num_samples=100)

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
# Average softmax probabilities over multiple stochastic forward passes.

num_mc = 50
with torch.no_grad():
    sample_probs = torch.stack(
        [dropout_model(X_test).softmax(-1) for _ in range(num_mc)]
    ).numpy()  # (num_mc, N, 10)
mean_probs = sample_probs.mean(0)

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
    title="Top-5 Most Uncertain Test Predictions (MC Dropout)",
)
plot.show()
