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
from torch.utils.data import DataLoader, TensorDataset

from probly.method.deup import deup
from probly.representer import representer

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# %%
# Wrap the base model with DEUP

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
# Phase 1 Training: Train the base classifier (encoder + classification head)

print("Phase 1 Training: Base Classifier")
optimizer_phase1 = torch.optim.Adam(
    list(deup_model.encoder.parameters()) + list(deup_model.classification_head.parameters()),
    lr=1e-3,
)

deup_model.train()
for epoch in range(150):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        features = deup_model.encoder(inputs)
        logits = deup_model.classification_head(features)
        loss = nn.functional.cross_entropy(logits, targets)

        optimizer_phase1.zero_grad()
        loss.backward()
        optimizer_phase1.step()

# %%
# Phase 2 Training: Fit stationarizing features and train the error head

print("\nPhase 2 Training: Error Head")

# Freeze the encoder and classification head
for param in deup_model.encoder.parameters():
    param.requires_grad = False
for param in deup_model.classification_head.parameters():
    param.requires_grad = False

# %%
# Fit stationarizing feature providers on in-distribution data only

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


ood_X = torch.FloatTensor(500, 2).uniform_(-3, 3)
ood_y = torch.randint(0, 2, (500,))
phase2_loader = DataLoader(
    torch.utils.data.ConcatDataset([dataset, torch.utils.data.TensorDataset(ood_X, ood_y)]),
    batch_size=64,
    shuffle=True,
)

# %%
# Train the error head

optimizer_phase2 = torch.optim.Adam(deup_model.error_head.parameters(), lr=1e-2)
mse_loss_fn = nn.MSELoss()

# %%
# Train

deup_model.train()
for epoch in range(250):
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
# Evaluate predictive uncertainty

deup_model.eval()
rep = representer(deup_model)

plot = plot_example_uncertainty(X, y, rep, title="DEUP Predictive Uncertainty")
plot.show()
