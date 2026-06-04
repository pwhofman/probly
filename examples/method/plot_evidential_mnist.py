"""=====================
Evidential on MNIST
=====================

Evidential Deep Learning replaces the softmax output with a Dirichlet
distribution, learning to predict the distribution over class probabilities
directly.  Uncertainty is high when evidence is spread across many classes
or concentrated on a class the model has not seen before.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.evidential import evidential_classification
from probly.metrics import roc_curve
from probly.plot import plot_histogram, plot_roc_curve
from probly.quantification import quantify
from probly.representer import representer
from probly.train.evidential.torch import evidential_log_loss, evidential_kl_divergence
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
evidential_model = evidential_classification(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# Train using the evidential log-loss, which combines MSE for the evidence
# and a KL-divergence term to regularize the distribution.
# The KL-weight is annealed over the first few epochs to allow the model
# to learn the evidence before enforcing the prior.

X_train_batches, y_train_batches = zip(*train_loader)
X_train_flat = torch.cat([x.view(-1, 28 * 28) for x in X_train_batches])
y_train = torch.cat(list(y_train_batches))

flat_dataloader = DataLoader(
    TensorDataset(X_train_flat, y_train),
    batch_size=256,
    shuffle=True,
)

opt = torch.optim.Adam(evidential_model.parameters(), lr=1e-3)
grad_clip_norm = 0.5

kl_weight = 0.01
annealing_epochs = 2

evidential_model.train()
for epoch in range(5):

    if annealing_epochs == 0:
        lambda_t = kl_weight
    else:
        lambda_t = kl_weight * min(1.0, epoch / annealing_epochs)

    for inputs, targets in flat_dataloader:
        opt.zero_grad()

        alpha = evidential_model(inputs)

        loss_val = evidential_log_loss(alpha, targets) + lambda_t * evidential_kl_divergence(alpha, targets)

        loss_val.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(evidential_model.parameters(), grad_clip_norm)

        opt.step()

# %%
# Uncertainty Quantification
# --------------------------

evidential_model.eval()
rep = representer(evidential_model, num_samples=200)

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
    out = evidential_model(X_test)
    logits = out[0] if isinstance(out, tuple) else out
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
    title="Top-5 Most Uncertain Test Predictions (Evidential)",
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
