"""=================
DEUP on Two Moons
=================

Direct epistemic uncertainty prediction trains a base classifier
and then a separate error head to explicitly predict per-sample
classification errors, using this predicted error score as a
direct measure of the model's uncertainty regarding its own knowledge.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from typing import Any

from probly.method.deup import deup
from probly.representer import representer

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# %%
# Model
# -----
#
# Wrap the base model with DEUP. We define stationarizing features that combine
# density estimates and dropout variance to create a robust input for the error head.
#
# The model is moved to the available device (GPU if present, otherwise CPU) to ensure
# efficient computation during training and density fitting, while maintaining compatibility
# across different hardware setups without manual code changes.

base_model = MLPClassifier()

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
# Phase 1: Training Base Classifier
# ---------------------------------
#
# Train the encoder and classification head with standard cross-entropy loss.
# Gradient clipping is applied to ensure stability.

opt_main = torch.optim.Adam(
    list(deup_model.encoder.parameters()) + list(deup_model.classification_head.parameters()),
    lr=1e-3
)
criterion = nn.CrossEntropyLoss()
grad_clip_norm = 0.5

deup_model.train()
for epoch in range(50):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt_main.zero_grad()

        features = deup_model.encoder(inputs)
        logits = deup_model.classification_head(features)
        loss = criterion(logits, targets)

        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(deup_model.parameters(), grad_clip_norm)
        opt_main.step()

# Freeze the encoder and classification head
for param in deup_model.encoder.parameters():
    param.requires_grad = False
for param in deup_model.classification_head.parameters():
    param.requires_grad = False

# %%
# Phase 2: Prepare Stationarizing Features & OOD Data
# ---------------------------------------------------
#
# 1. Fit auxiliary providers (e.g., GMM, Dropout) to compute stationarizing features.
# 2. Generate synthetic OOD data (uniform noise) to teach the error head about
#    high-error regions.
# 3. Compute features and target error values (log of BCE loss) for all data.

providers: list[Any] = list(getattr(deup_model, "providers", []))

for provider in providers:
    provider.to(device)
    provider.fit(deup_model.encoder, deup_model.classification_head, train_loader, device, False)

_orig_phi = deup_model._compute_stationarizing_features
deup_model._compute_stationarizing_features = lambda *a: _orig_phi(*a).clamp(-10.0, 10.0)

# Augment the in-distribution data with synthetic uniform-noise OOD so the
# error head sees high-error regions and learns to flag off-manifold inputs.
ood_X = torch.FloatTensor(500, 2).uniform_(-3, 3)
ood_y = torch.randint(0, 2, (500,))
phase2_loader = DataLoader(
    ConcatDataset([dataset, TensorDataset(ood_X, ood_y)]),
    batch_size=64,
    shuffle=True,
)

bce_criterion = nn.BCELoss(reduction="none")
deup_model.eval()

all_phi: list[torch.Tensor] = []
all_targets: list[torch.Tensor] = []

with torch.no_grad():
    for inputs_, targets_ in phase2_loader:
        inputs, targets = inputs_.to(device), targets_.to(device)

        features = deup_model.encoder(inputs)
        logits = deup_model.classification_head(features)

        phi = deup_model._compute_stationarizing_features(features, logits)  # noqa: SLF001

        probs = torch.softmax(logits.float(), dim=-1).detach().cpu()
        one_hot = nn.functional.one_hot(targets_, num_classes=probs.size(-1)).float().detach().cpu()
        per_sample_bce = bce_criterion(probs, one_hot).sum(dim=-1)

        target_val = torch.log10(per_sample_bce.clamp(min=1e-10)).clamp(min=-5.0)

        all_phi.append(phi.detach().cpu())
        all_targets.append(target_val.detach().cpu())

phi_all = torch.cat(all_phi)
targets_all = torch.cat(all_targets)

error_head_dataset = torch.utils.data.TensorDataset(phi_all, targets_all)
error_head_loader = torch.utils.data.DataLoader(error_head_dataset, batch_size=64, shuffle=True, drop_last=False)

# %%
# Phase 3: Training Error Head
# -------------------------
#
# Train the error head to predict the target error values from the stationarizing features.
# This head acts as the final uncertainty estimator.

deup_model.error_head.to(device)
opt_error = torch.optim.SGD(deup_model.error_head.parameters(), lr=0.005, momentum=0.9)
mse_loss_fn = nn.MSELoss()

deup_model.error_head.train()
for epoch in range(200):
    for phi_batch, tgt_batch in error_head_loader:
        phi, tgt = phi_batch.to(device, non_blocking=True), tgt_batch.to(device, non_blocking=True)

        loss = nn.functional.mse_loss(deup_model.error_head(phi), tgt)
        opt_error.zero_grad()
        loss.backward()
        opt_error.step()

# %%
# Evaluation
# ----------

deup_model.error_head.eval()

rep = representer(deup_model)
plot = plot_example_uncertainty(X, y, rep, title="DEUP Predictive Uncertainty")
plot.show()
