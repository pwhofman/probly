"""=======================
Credal Net on MNIST
=======================

Credal Net applies interval arithmetic through the network layers to propagate
weight uncertainty forward from inputs to predictions.  The resulting credal
set represents the range of possible class probabilities consistent with the
model's imprecision, producing probability intervals for each class.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.credal_net import credal_net
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
from probly.predictor import predict_raw
from probly.quantification import quantify
from probly.representer import representer
from probly.train.credal.torch import intersection_probability_ce_loss
from probly.utils.torch import intersection_probability
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
# Wrap a base classifier with ``credal_net`` so each weight becomes a learnable
# interval; ``use_base_weights=True`` initializes the interval centers from the
# (untrained) base weights.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
prob_model = nn.Sequential(base_model, nn.Softmax(dim=1))
credal_model = credal_net(
    prob_model,
    predictor_type="probabilistic_classifier",
    use_base_weights=True,
)

# %%
# Training
# --------
#
# Train the wrapped credal net directly with the intersection-probability
# cross-entropy loss (Eq. 14 of :cite:`wang2024credalnet`), which operates on
# the packed ``(lower, upper)`` interval output produced by ``predict_raw``.

opt = torch.optim.Adam(credal_model.parameters(), lr=1e-3)

credal_model.train()
for _epoch in range(5):
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        output = predict_raw(credal_model, X_flat)
        loss = intersection_probability_ce_loss(output, y_batch)
        loss.backward()
        opt.step()
credal_model.eval()

# %%
# Uncertainty Quantification
# --------------------------

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
#
# The point prediction is the intersection probability of the learned
# ``(lower, upper)`` class-probability interval.

with torch.no_grad():
    raw = predict_raw(credal_model, X_test)
    n_classes = raw.shape[-1] // 2
    mean_probs = intersection_probability(raw[..., :n_classes], raw[..., n_classes:]).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (Credal Net)",
)
plot.show()

# %%
# OOD Detection
# -------------
#
# Fashion MNIST shares MNIST's image format but contains clothing categories
# the model was never trained on.  The same representer from the UQ section is
# applied to the OOD data -- no new inference setup needed.

ood_dataset = datasets.FashionMNIST(
    "~/.cache/mnist", train=False, download=True, transform=transforms.ToTensor()
)
X_ood = torch.stack([ood_dataset[i][0].flatten() for i in range(len(ood_dataset))])

with torch.no_grad():
    credal_set_ood = rep.represent(X_ood)

uq_ood = quantify(credal_set_ood)
_unc_ood = uq_ood.total if hasattr(uq_ood, "total") else (uq_ood.epistemic if hasattr(uq_ood, "epistemic") else uq_ood.aleatoric)
uncertainty_ood = _unc_ood.detach().numpy() if isinstance(_unc_ood, torch.Tensor) else np.asarray(_unc_ood)
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
    title="Upper Entropy: MNIST (ID) vs Fashion MNIST (OOD)",
)
fig.axes[0].set_xlabel("Upper Entropy (bits)")
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
