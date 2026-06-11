"""==============================
Credal Ensembling on MNIST
==============================

Credal Ensembling trains an ensemble of classifiers and represents their
joint predictions as a Convex Credal Set -- the convex hull of all member
probability vectors.  The set grows when members disagree, so the upper
entropy (total uncertainty) rises on ambiguous or out-of-distribution inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.credal_ensembling import credal_ensembling
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

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
credal_model = credal_ensembling(
    base_model,
    predictor_type="logit_classifier",
    num_members=5,
)

# %%
# Training
# --------
#
# Each member is trained independently to maximize convex hull volume.

for member in credal_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for _epoch in range(5):
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_flat = X_batch.view(-1, 28 * 28)
            opt.zero_grad()
            logits = member(X_flat)
            loss = nn.functional.cross_entropy(logits, y_batch)
            loss.backward()
            opt.step()
            correct += (logits.detach().argmax(-1) == y_batch).sum().item()
            total += len(y_batch)
        if correct / total >= 0.97:
            break
    member.eval()

# %%
# Uncertainty Quantification
# --------------------------
#
# The representer builds the convex hull credal set from all member predictions.
# ``quantify`` returns a ``CredalSetEntropyDecomposition``; total uncertainty
# is the upper entropy of the convex hull.

rep = representer(credal_model)

with torch.no_grad():
    credal_set = rep.represent(X_test)

uq = quantify(credal_set)
_unc = uq.total if hasattr(uq, "total") else (uq.epistemic if hasattr(uq, "epistemic") else uq.aleatoric)
uncertainty = _unc.detach().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    member_probs = torch.stack([member(X_test).softmax(-1) for member in credal_model]).numpy()
mean_probs = member_probs.mean(0)

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
    title="Top-5 Most Uncertain Test Predictions (Credal Ensembling)",
)
plot.show()
