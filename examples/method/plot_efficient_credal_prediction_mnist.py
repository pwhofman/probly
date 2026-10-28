"""======================================
Efficient Credal Prediction on MNIST
======================================

Efficient credal prediction turns a single trained classifier into a credal
predictor. After training, it determines for each class the largest additive
offsets of that class's logit under which the relative likelihood of the
training data stays above a threshold ``alpha``. At inference time, perturbing
the logits by these offsets yields probability intervals, whose width reflects
the epistemic uncertainty of the prediction. The appeal of the method lies in
its cost: only one model is trained, and the bounds are calibrated post hoc
from its training logits.

This example applies the method to MNIST and shows the test digits on which
the resulting credal predictor is most uncertain.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.method.efficient_credal_prediction import (
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction,
)
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
ecp = efficient_credal_prediction(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# The wrapper leaves the base network unchanged, so the model is trained like
# any ordinary classifier with the cross-entropy loss.

opt = torch.optim.Adam(ecp.parameters(), lr=1e-3)
ecp.train()
for _epoch in range(5):
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(ecp(X_batch.view(-1, 28 * 28)), y_batch)
        loss.backward()
        opt.step()
ecp.eval()

# %%
# Post-hoc Calibration of the Bounds
# ----------------------------------
#
# The per-class offsets for the threshold ``alpha`` are computed from the
# logits on the training data and stored on the predictor as ``lower`` and
# ``upper``.
#
# The choice of ``alpha`` requires more care here than in low-dimensional toy
# problems. A well-fit MNIST classifier produces large training logits, so
# shifting a single class logit changes the training likelihood only slightly,
# and a loose threshold consequently admits very large offsets. With
# ``alpha=0.5``, for instance, hundreds of test images would receive the
# maximal uncertainty of ``log2(10)`` bits. A strict threshold such as
# ``alpha=0.99`` keeps the credal sets informative.

with torch.no_grad():
    logits_train, targets_train = zip(
        *((ecp(X_batch.view(-1, 28 * 28)), y_batch) for X_batch, y_batch in train_loader)
    )
logits_train = torch.cat(logits_train)
targets_train = torch.cat(targets_train)

lower, upper = compute_efficient_credal_prediction_bounds(
    logits_train, targets_train, num_classes=10, alpha=0.99
)
ecp.lower, ecp.upper = lower.float(), upper.float()

# %%
# Uncertainty Quantification
# --------------------------
#
# The representer turns the base logits and the calibrated offsets into a
# credal set per test image. Its uncertainty is quantified with ``quantify``
# and converted from nats to bits.

rep = representer(ecp)

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
#
# Point predictions are the softmax probabilities of the unperturbed base
# classifier.

with torch.no_grad():
    mean_probs = ecp(X_test).softmax(-1).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (Efficient Credal Prediction)",
)
plot.show()
