"""=======================
Credal BNN on MNIST
=======================

The Credal BNN trains a Bayesian ensemble whose members sample different
posterior modes, producing a Sample-Mean Convex Credal Set (SMCCS).
Wide credal sets indicate high epistemic uncertainty -- disagreement between
posterior samples about the correct probability distribution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.credal_bnn import credal_bnn
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
from probly.quantification import quantify
from probly.representer import representer
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence
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
credal_model = credal_bnn(
    base_model,
    predictor_type="logit_classifier",
    num_members=5,
)

# %%
# Training
# --------
#
# Train each member with the ELBO objective: cross-entropy on the logits plus a
# KL penalty on the variational posterior, mirroring the benchmark recipe.

criterion = ELBOLoss(kl_penalty=1e-5)

for member in credal_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for _epoch in range(5):
        for X_batch, y_batch in train_loader:
            X_flat = X_batch.view(-1, 28 * 28)
            opt.zero_grad()
            logits = member(X_flat)
            kl = collect_kl_divergence(member)
            loss = criterion(logits, y_batch, kl)
            loss.backward()
            opt.step()
    member.eval()

# %%
# Uncertainty Quantification
# --------------------------
#
# The representer builds an SMCCS from all member predictions.
# ``quantify`` returns a ``CredalSetEntropyDecomposition`` where total
# uncertainty is the upper entropy of the credal set.

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
    title="Top-5 Most Uncertain Test Predictions (Credal BNN)",
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
