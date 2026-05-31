"""=====================================
Bayesian Neural Network on MNIST
=====================================

Replace point-estimate weights with distributions and train them with the ELBO loss.
Every forward pass samples new weights, so predictions are inherently stochastic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
from probly.quantification import quantify
from probly.representer import representer
from probly.transformation import bayesian
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

N_train = len(train_loader.dataset)

# %%
# Model
# -----

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

bayesian_model = bayesian(
    base_model,
    use_base_weights=False,  # initialize posterior means randomly rather than from base_model
    posterior_std=0.05,      # initial posterior std; small = near-deterministic start
    prior_mean=0.0,
    prior_std=1.0,           # smaller = stronger regularization toward zero
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# ELBOLoss(beta) computes: cross_entropy(out, y) + beta * kl.
# beta = 1/N scales the KL so its magnitude is independent of dataset size.
# collect_kl_divergence walks the model and sums the KL from every
# BayesianLinear layer, which must be called after each forward pass because
# each forward pass draws new weight samples.

opt = torch.optim.Adam(bayesian_model.parameters(), lr=1e-3)
criterion = ELBOLoss(1.0 / N_train)

bayesian_model.train()
for _epoch in range(10):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        out = bayesian_model(X_batch.view(-1, 28 * 28))
        kl = collect_kl_divergence(bayesian_model)
        loss = criterion(out, y_batch, kl)
        loss.backward()
        opt.step()
        correct += (out.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.97:
        break

# %%
# Uncertainty Quantification
# --------------------------

bayesian_model.eval()
rep = representer(bayesian_model, num_samples=100)

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
# Average softmax probabilities over multiple Bayesian weight samples.

num_mc = 50
with torch.no_grad():
    sample_probs = torch.stack(
        [bayesian_model(X_test).softmax(-1) for _ in range(num_mc)]
    ).numpy()  # (num_mc, N, 10)
mean_probs = sample_probs.mean(0)

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
    title="Top-5 Most Uncertain Test Predictions (Bayesian Neural Network)",
)
plot.show()

# %%
# OOD Detection
# -------------
#
# Fashion MNIST shares MNIST's image format (28x28 grayscale) but contains
# clothing categories the model was never trained on.  The same representer
# from the UQ section is applied to the OOD data -- no new inference setup needed.

ood_dataset = datasets.FashionMNIST(
    "~/.cache/mnist", train=False, download=True, transform=transforms.ToTensor()
)
X_ood = torch.stack([ood_dataset[i][0].flatten() for i in range(len(ood_dataset))])

with torch.no_grad():
    representation_ood = rep.represent(X_ood)

uq_ood = quantify(representation_ood)
_total_ood = uq_ood.total
uncertainty_ood = (
    _total_ood.detach().numpy() if isinstance(_total_ood, torch.Tensor) else np.asarray(_total_ood)
)
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
    title="Predictive Entropy: MNIST (ID) vs Fashion MNIST (OOD)",
)
fig.axes[0].set_xlabel("Predictive Entropy (bits)")
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
