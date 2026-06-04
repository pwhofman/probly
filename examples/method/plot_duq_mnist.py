"""===============
DUQ on MNIST
===============

Deep Uncertainty Quantification (DUQ) replaces the softmax head with a radial
basis function (RBF) network that maps feature representations to per-class
centroids.  Uncertainty is estimated from the kernel distances between an
input's representation and the learned centroids.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.duq import duq
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
duq_model = duq(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# DUQ uses binary cross-entropy on the kernel outputs together with a gradient
# penalty that enforces a bi-Lipschitz constraint on the feature map.

opt = torch.optim.Adam(duq_model.parameters(), lr=1e-3)
criterion = nn.BCELoss(reduction = "mean")

gradient_penalty = 0.5
num_classes = 10

duq_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for X_batch, y_batch in train_loader:
        X_flat = X_batch.view(-1, 28 * 28).detach().requires_grad_(True)
        targets_onehot = F.one_hot(y_batch, num_classes).float()


        kernel_values = duq_model(X_flat)
        loss = criterion(kernel_values, targets_onehot)

        gradients = torch.autograd.grad(
            outputs=kernel_values,
            inputs=X_flat,
            grad_outputs=torch.ones_like(kernel_values),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_norm = gradients.flatten(start_dim=1).norm(2, dim=1)
        duq_penalty = ((grad_norm - 1.0) ** 2).mean()
        total_loss = loss + gradient_penalty * duq_penalty

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        correct += (kernel_values.detach().argmax(-1) == y_batch).sum().item()
        total += len(y_batch)
    if correct / total >= 0.95:
        break

# %%
# Uncertainty Quantification
# --------------------------

duq_model.eval()
rep = representer(duq_model)

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
#
# The DUQ kernel outputs are already in [0, 1] and interpreted as class scores.

with torch.no_grad():
    kernel_values = duq_model(X_test)
    mean_probs = (kernel_values / kernel_values.sum(-1, keepdim=True)).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (DUQ)",
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
