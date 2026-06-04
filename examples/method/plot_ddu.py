"""===========================================
Deep Deterministic Uncertainty on Two Moons
===========================================

Deep Deterministic Uncertainty (DDU) applies spectral normalization to a
feature extractor, then fits a class-conditional Gaussian density model on
the training features.  A single deterministic forward pass yields both
a class prediction and a feature-density score for epistemic uncertainty.
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
# DDU wraps the backbone with spectral normalization (controlled by ``sn_coeff``)
# to smooth the Lipschitz constant of the feature map, which is required for
# the density score to be a reliable distance proxy.

base_model = MLPClassifier()

ddu_model = ddu(base_model, sn_coeff=3.0, predictor_type="logit_classifier")



# %%
# Training
# --------

opt = torch.optim.Adam(ddu_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

ddu_model.train()
for epoch in range(200):
    opt.zero_grad()


    features = ddu_model.encoder(X_tensor)
    logits = ddu_model.classification_head(features)


    loss = criterion(logits, y_tensor)

    loss.backward()
    opt.step()

# %%
# Fit Density Head
# ----------------
#
# Collect all training features in one pass and fit the class-conditional
# Gaussians.  This only needs to happen once after training.


ddu_model.eval()

with torch.no_grad():
    ddu_model.fit_density_head(X_tensor, y_tensor)

# %%
# Uncertainty Evaluation
# ----------------------

ddu_model.eval()

rep = representer(ddu_model)

plot = plot_example_uncertainty(X, y, rep, title="DDU Predictive Uncertainty")
plot.show()
