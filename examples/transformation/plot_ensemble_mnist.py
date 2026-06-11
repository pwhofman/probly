"""============================
Deep Ensemble on MNIST
============================

Train several independent copies of the same network from different initializations and average their predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.quantification import quantify
from probly.representer import representer
from probly.transformation import ensemble
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

ensemble_model = ensemble(
    base_model,
    num_members=3,
    reset_params=True,  # fresh init per member maximizes diversity
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Train each member independently with standard cross-entropy.
# The uncertainty signal comes from disagreement between members,
# not from the loss itself.

ensemble_model.train()
for member in ensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
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

ensemble_model.eval()
rep = representer(ensemble_model)

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
# Collect per-member softmax probabilities and their mean.

with torch.no_grad():
    member_probs = torch.stack(
        [member(X_test).softmax(dim=-1) for member in ensemble_model]
    ).numpy()  # (num_members, N, 10)
mean_probs = member_probs.mean(0)

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------
#
# Plot the five most uncertain test digits with per-member probability lines.

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (Deep Ensemble)",
)
plot.show()
