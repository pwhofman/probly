"""================
DEUP on MNIST
================

Direct Epistemic Uncertainty Prediction (DEUP) trains a base classifier in
phase one, then trains a separate error head in phase two that explicitly
predicts per-sample cross-entropy errors using stationarizing features.
The predicted error score serves as a direct measure of epistemic uncertainty.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from torchvision import datasets, transforms

from probly.evaluation.ood import evaluate_ood
from probly.method.deup import deup
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

X_train_batches, y_train_batches = zip(*train_loader)
X_train = torch.cat([x.view(-1, 28 * 28) for x in X_train_batches])
y_train = torch.cat(list(y_train_batches))
train_dataset_flat = TensorDataset(X_train, y_train)
train_loader_flat = DataLoader(train_dataset_flat, batch_size=256, shuffle=True)

# %%
# Model
# -----

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)

deup_model = deup(
    base_model,
    hidden_size=512,
    n_hidden_layers=2,
    stationarizing_features=[
        "log_gmm_density",
        "log_mc_dropout_variance",
    ],
    predictor_type="logit_classifier",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deup_model.to(device)

# %%
# Phase 1 Training: Base Classifier
# -----------------------------------
#
# Train the encoder and classification head with standard cross-entropy.

print("Phase 1: Training base classifier")
optimizer_phase1 = torch.optim.Adam(
    list(deup_model.encoder.parameters()) + list(deup_model.classification_head.parameters()),
    lr=1e-3,
)

deup_model.train()
for _epoch in range(5):
    correct, total = 0, 0
    for inputs, targets in train_loader_flat:
        inputs, targets = inputs.to(device), targets.to(device)
        features = deup_model.encoder(inputs)
        logits = deup_model.classification_head(features)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer_phase1.zero_grad()
        loss.backward()
        optimizer_phase1.step()
        correct += (logits.detach().argmax(-1) == targets).sum().item()
        total += len(targets)
    if correct / total >= 0.97:
        break

# %%
# Phase 2 Training: Error Head
# -----------------------------
#
# Freeze the backbone, fit stationarizing feature providers on in-distribution
# data, then train the error head on both in-distribution and OOD (uniform
# noise) inputs.

print("\nPhase 2: Training error head")

for param in deup_model.encoder.parameters():
    param.requires_grad = False
for param in deup_model.classification_head.parameters():
    param.requires_grad = False

deup_model.eval()
for provider in deup_model.providers:
    provider.fit(
        deup_model.encoder,
        deup_model.classification_head,
        train_loader_flat,
        device,
    )

_orig_phi = deup_model._compute_stationarizing_features
# clamp stationarizing features to prevent exploding inputs to the error head
deup_model._compute_stationarizing_features = lambda *a: _orig_phi(*a).clamp(-10.0, 10.0)

ood_X = torch.FloatTensor(len(X_train), 28 * 28).uniform_(0, 1)
ood_y = torch.randint(0, 10, (len(X_train),))
phase2_loader = DataLoader(
    ConcatDataset([train_dataset_flat, TensorDataset(ood_X, ood_y)]),
    batch_size=256,
    shuffle=True,
)

optimizer_phase2 = torch.optim.Adam(deup_model.error_head.parameters(), lr=1e-2)
mse_loss_fn = nn.MSELoss()

deup_model.train()
for _epoch in range(5):
    for inputs, targets in phase2_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            features = deup_model.encoder(inputs)
            logits = deup_model.classification_head(features)
            per_sample_ce = nn.functional.cross_entropy(logits, targets, reduction="none")
            log10_ce_target = torch.log10(torch.clamp(per_sample_ce, min=1e-6, max=1e4))
            stationarizing_features = deup_model._compute_stationarizing_features(features, logits)
        predicted_log10_error = deup_model.error_head(stationarizing_features)
        loss = mse_loss_fn(predicted_log10_error, log10_ce_target)
        optimizer_phase2.zero_grad()
        loss.backward()
        optimizer_phase2.step()

# %%
# Uncertainty Quantification
# --------------------------

deup_model.eval()
rep = representer(deup_model)

X_test_dev = X_test.to(device)
with torch.no_grad():
    representation = rep.represent(X_test_dev)

uq = quantify(representation)
_unc = uq.total if hasattr(uq, "total") else (uq.epistemic if hasattr(uq, "epistemic") else uq.aleatoric)
uncertainty = _unc.detach().cpu().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    features = deup_model.encoder(X_test_dev)
    mean_probs = deup_model.classification_head(features).softmax(-1).cpu().numpy()

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
    title="Top-5 Most Uncertain Test Predictions (DEUP)",
    unit="score",
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
X_ood = torch.stack([ood_dataset[i][0].flatten() for i in range(len(ood_dataset))]).to(device)

with torch.no_grad():
    representation_ood = rep.represent(X_ood)

uq_ood = quantify(representation_ood)
_unc_ood = uq_ood.total if hasattr(uq_ood, "total") else (uq_ood.epistemic if hasattr(uq_ood, "epistemic") else uq_ood.aleatoric)
uncertainty_ood = _unc_ood.detach().cpu().numpy() if isinstance(_unc_ood, torch.Tensor) else np.asarray(_unc_ood)
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
    title="Predicted Error Score: MNIST (ID) vs Fashion MNIST (OOD)",
)
fig.axes[0].set_xlabel("Predicted Error Score")
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
