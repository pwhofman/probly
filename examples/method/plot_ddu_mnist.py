"""===============
DDU on MNIST
===============

Deep Deterministic Uncertainty (DDU) applies spectral normalization to a
feature extractor, then fits a class-conditional Gaussian density model on
the training features.  A single deterministic forward pass yields both
a class prediction and a feature-density score for epistemic uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.ddu import ddu
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.data import load_mnist

from examples.utils.model import ResFFN
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
# DDU wraps the backbone with spectral normalization (controlled by ``sn_coeff``)
# to smooth the Lipschitz constant of the feature map, which is required for
# the density score to be a reliable distance proxy.

base_model = ResFFN(in_features = 28*28, hidden_features = 256, out_features = 10)

ddu_model = ddu(base_model, sn_coeff=5.0, predictor_type="logit_classifier")

# %%
# Training
# --------

opt = torch.optim.Adam(ddu_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

ddu_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        features = ddu_model.encoder(X_flat)
        logits = ddu_model.classification_head(features)

        loss = criterion(logits, y_batch)

        loss.backward()
        opt.step()
        correct += (logits.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Fit Density Head
# ----------------
#
# Collect all training features in one pass and fit the class-conditional
# Gaussians.  This only needs to happen once after training.

ddu_model.eval()

all_features = []
all_labels = []

with torch.no_grad():
    for inputs, targets in train_loader:
        if inputs.dim() == 4:
            inputs_flat = inputs.view(inputs.size(0), -1)
        else:
            inputs_flat = inputs

        features = ddu_model.encoder(inputs_flat)
        all_features.append(features.detach().cpu())
        all_labels.append(targets.detach().cpu())

features_cat = torch.cat(all_features)
labels_cat = torch.cat(all_labels)

density_head = ddu_model.density_head
density_head.fit(features_cat, labels_cat)

# %%
# Uncertainty Quantification
# --------------------------

rep = representer(ddu_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_unc = uq.total if hasattr(uq, "total") else (uq.epistemic if hasattr(uq, "epistemic") else uq.aleatoric)
uncertainty = _unc.detach().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    out = ddu_model(X_test)
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
    title="Top-5 Most Uncertain Test Predictions (DDU)",
    unit="nats",
)
plot.show()
