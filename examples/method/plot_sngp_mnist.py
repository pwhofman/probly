"""================
SNGP on MNIST
================

Spectral-normalized Neural Gaussian Process (SNGP) replaces the final dense
layer with a Gaussian Process approximation.  Spectral normalization preserves
input-space distance in the feature map, so the GP posterior variance grows
smoothly as inputs move away from the training distribution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.sngp import reset_precision_matrix, sngp
from probly.quantification import quantify
from probly.representer import representer
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
# SNGP wraps the backbone with spectral normalization and replaces the linear
# output head with a random Fourier feature approximation to a GP.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

sngp_model = sngp(
    base_model,
    num_random_features=1024,
    ridge_penalty=0.01,
    norm_multiplier=0.9,
    n_power_iterations=1,
)
opt = torch.optim.Adam(sngp_model.parameters(), lr=1e-3)

# %%
# Training
# --------
#
# The GP precision matrix is reset at the start of every epoch so it
# accumulates statistics across the full training set.  The loss is
# cross-entropy on the GP MAP logits returned by the model.

sngp_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    reset_precision_matrix(sngp_model)
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        out = sngp_model(X_flat)
        logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
        loss = nn.functional.cross_entropy(logits, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        correct += (logits.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------

sngp_model.eval()
rep = representer(sngp_model, num_samples=800)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_unc = uq.total if hasattr(uq, "total") else (uq.epistemic if hasattr(uq, "epistemic") else uq.aleatoric)
uncertainty = _unc.detach().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    out = sngp_model(X_test)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
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
    title="Top-5 Most Uncertain Test Predictions (SNGP)",
)
plot.show()
