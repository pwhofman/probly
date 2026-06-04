"""=================
DEUP on Two Moons
=================

Direct Epistemic Uncertainty Prediction (DEUP) trains a base classifier in
phase one, then trains a separate error head in phase two that explicitly
predicts per-sample cross-entropy errors using stationarizing features.
The predicted error score serves as a direct measure of epistemic uncertainty.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

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

optimizer_phase1 = torch.optim.Adam(
    list(deup_model.encoder.parameters()) + list(deup_model.classification_head.parameters()),
    lr=1e-3
)
criterion = nn.CrossEntropyLoss()

deup_model.train()
for epoch in range(50):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        features = deup_model.encoder(inputs)
        logits = deup_model.classification_head(features)
        loss = criterion(logits, targets)

        optimizer_phase1.zero_grad()
        loss.backward()
        optimizer_phase1.step()

# %%
# Phase 2: Prepare Stationarizing Features & OOD Data
# ---------------------------------------------------
#
# Collecting the DEUP error targets.
#
# 1. Freeze the backbone.
# 2. Fit auxiliary providers (e.g., GMM, Dropout) to compute stationarizing features.
# 3. Generate synthetic OOD data (uniform noise) to teach the error head about
#    high-error regions.
# 4. Compute features and target error values (log of BCE loss) for all data.

for param in deup_model.encoder.parameters():
    param.requires_grad = False
for param in deup_model.classification_head.parameters():
    param.requires_grad = False


deup_model.eval()
for provider in deup_model.providers:
    provider.fit(
        deup_model.encoder,
        deup_model.classification_head,
        train_loader,
        device,
    )

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

all_phi: list[torch.Tensor] = []
all_targets: list[torch.Tensor] = []

deup_model.eval()
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
# ----------------------------
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
