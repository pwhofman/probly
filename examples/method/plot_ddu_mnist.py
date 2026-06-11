"""===============
DDU on MNIST
===============

Deep Deterministic Uncertainty (DDU) applies spectral normalization to a
feature extractor, then fits a class-conditional Gaussian density model on
the training features.  A single deterministic forward pass yields both
a class prediction and a feature-density score for epistemic uncertainty.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.ddu import ddu
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
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
    representation_ood = rep.represent(X_ood)

uq_ood = quantify(representation_ood)
_unc_ood = uq_ood.total if hasattr(uq_ood, "total") else (uq_ood.epistemic if hasattr(uq_ood, "epistemic") else uq_ood.aleatoric)
uncertainty_ood = _unc_ood.detach().numpy() if isinstance(_unc_ood, torch.Tensor) else np.asarray(_unc_ood)
if uncertainty_ood.ndim > 1:
    uncertainty_ood = uncertainty_ood.sum(axis=-1)

# %%
# Score Histogram
# ---------------
#
# The ID distribution should be tightly concentrated near zero (high confidence);
# the OOD distribution should shift right (higher uncertainty score).

fig = plot_histogram(
    uncertainty,
    uncertainty_ood,
    title="DDU Feature Density: MNIST (ID) vs Fashion MNIST (OOD)",
)
fig.axes[0].set_xlabel("Log-Density Score (nats)")
plt.show()

# %%
# ROC Curve
# ---------
#
# AUROC measures how well the uncertainty score separates ID from OOD samples
# end-to-end.  A perfect detector scores 1.0; random guessing scores 0.5.

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
