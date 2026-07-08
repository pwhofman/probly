"""==================
Laplace on MNIST
==================

The Laplace Approximation is a post-hoc method that turns a trained neural
network into a Bayesian model by fitting a Gaussian over the last-layer weights.
Uncertainty concentrates on inputs far from the training distribution.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from laplace import Laplace
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
# Train a standard MLP classifier to convergence; the Laplace approximation
# is applied afterwards as a post-hoc uncertainty wrapper.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
flat_model = nn.Sequential(nn.Flatten(), base_model)

opt = torch.optim.Adam(flat_model.parameters(), lr=1e-3)

flat_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        out = flat_model(X_batch)
        loss = nn.functional.cross_entropy(out, y_batch)
        loss.backward()
        opt.step()
        correct += (out.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Laplace Approximation
# ---------------------
#
# Fit a Kronecker-factored (KFAC) Laplace approximation over the last layer of
# the trained model.  No retraining is needed.

flat_model.eval()

laplace_model = Laplace(
    flat_model,
    "classification",
    subset_of_weights="last_layer",
    hessian_structure="kron",
)
laplace_model.fit(train_loader)

# %%
# Uncertainty Quantification
# --------------------------

rep = representer(laplace_model, num_samples=200)

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
    mean_probs = flat_model(X_test).softmax(-1).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (Laplace)",
)
plot.show()
