"""=====================================
Credal Relative Likelihood on MNIST
=====================================

Credal Relative Likelihood builds an ensemble of perturbed classifiers whose
weight perturbations are bounded by a relative likelihood ratio.  The resulting
probability intervals represent the set of posteriors that are plausible given
the training evidence, quantifying epistemic uncertainty via set width.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
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
num_members = 5
credal_model = credal_relative_likelihood(
    base_model,
    predictor_type="logit_classifier",
    num_members=num_members,
)
members = list(credal_model)


def _train_one_epoch(member: torch.nn.Module, lr: float) -> None:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=lr)
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(member(X_flat), y_batch)
        loss.backward()
        opt.step()


@torch.no_grad()
def _log_likelihood(member: torch.nn.Module) -> float:
    member.eval()
    total, count = 0.0, 0
    for X_batch, y_batch in train_loader:
        log_probs = nn.functional.log_softmax(member(X_batch.view(-1, 28 * 28)), dim=-1)
        total += log_probs.gather(1, y_batch.unsqueeze(1)).sum().item()
        count += y_batch.numel()
    return total / count


# %%
# Training
# --------
#
# The first member is trained to convergence on the full data; each subsequent
# member is trained only until its relative likelihood reaches a per-member
# threshold, mirroring the benchmark training recipe.

for _epoch in range(5):
    _train_one_epoch(members[0], lr=1e-3)
max_ll = _log_likelihood(members[0])

alpha = 0.5
thresholds = torch.linspace(alpha, 1.0, num_members)[:-1].tolist()

for member, threshold in zip(members[1:], thresholds, strict=True):
    for _epoch in range(5):
        _train_one_epoch(member, lr=1e-3)
        rel_lik = float(np.exp(_log_likelihood(member) - max_ll))
        if rel_lik >= threshold:
            break

for member in members:
    member.eval()

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

with torch.no_grad():
    member_probs = torch.stack([
        member(X_test).softmax(-1) for member in credal_model
    ]).numpy()
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
    title="Top-5 Most Uncertain Test Predictions (Credal Relative Likelihood)",
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
