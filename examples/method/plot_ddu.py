"""===========================================
Deep Deterministic Uncertainty on Two Moons
===========================================

Deep Deterministic Uncertainty (DDU) combines spectral normalization
with a deep ensemble of feature extractors to fit a class-conditional
Gaussian density model, enabling the separation of epistemic and aleatoric
uncertainty via a single deterministic forward pass.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

import torch
from torch import nn

from probly.representer import representer
from probly.method.ddu import ddu

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
# Instantiate a standard MLP classifier and wrap it with the DDU method.
# The 'sn_coeff' parameter (set to 3.0 here) controls the strength of Spectral Normalization,
# which constrains the Lipschitz constant of the feature extractor. This prevents the model
# from becoming overly confident on out-of-distribution data by smoothing the feature space.
#
# The model is moved to the available device (GPU if present, otherwise CPU) to ensure
# efficient computation during training and density fitting, while maintaining compatibility
# across different hardware setups without manual code changes.

base_model = MLPClassifier()

ddu_model = ddu(base_model, sn_coeff=3.0, predictor_type="logit_classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddu_model.to(device)


# %%
# Training
# --------

opt = torch.optim.Adam(ddu_model.parameters(), lr=1e-3)

ddu_model.train()
for epoch in range(200):
    opt.zero_grad()

    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    features = ddu_model.encoder(X_tensor)
    logits = ddu_model.classification_head(features)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y_tensor)

    loss.backward()
    opt.step()

# %%
# Fitting Density Head
# --------------------
#
# After training, freeze the encoder and collect feature representations for all training samples.
# These features are used to fit a class-conditional Gaussian Mixture Model (the density head).
# This step is unique to DDU: it models the distribution of features for each class explicitly.
# At inference, uncertainty is derived from how likely a new feature vector is under these
# learned distributions, allowing the separation of epistemic (model) and aleatoric (data) uncertainty.

ddu_model.eval()
all_features: list[torch.Tensor] = []
all_labels: list[torch.Tensor] = []

with torch.no_grad():
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        features = ddu_model.encoder(inputs)
        all_features.append(features.cpu())
        all_labels.append(targets.cpu())

features_cat = torch.cat(all_features)
labels_cat = torch.cat(all_labels)

density_head = ddu_model.density_head
density_head_device = density_head.means.device
density_head.to(density_head_device)
density_head.fit(features_cat, labels_cat)

# %%
# Uncertainty Evaluation
# ----------------------

ddu_model.eval()

rep = representer(ddu_model)

plot = plot_example_uncertainty(X, y, rep, title="DDU Predictive Uncertainty")
plot.show()
