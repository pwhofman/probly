"""========================
Credal BNN Visualization
========================

This example trains a Bayesian ensemble using ``credal_bnn`` and renders
the resulting Sample-Mean Convex Credal Sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from probly.method.credal_bnn import credal_bnn
from probly.plot.credal import plot_credal_set
from probly.representer import representer
from probly.train.bayesian.torch import ELBOLoss, collect_kl_divergence

from examples.utils.model import MLPClassifier

np.random.seed(42)
torch.manual_seed(42)

# %%
# Setup
# -----
#
# Three well-separated Gaussian blobs, sampled around the simplex vertices,
# give a clean 3-class signal that makes the credal set geometry easy to read.

centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Model
# -----
#
# Wrap the base classifier with ``credal_bnn``: each ensemble member is an
# independent Bayesian neural network with its own posterior over the weights.

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)
credal_model = credal_bnn(base_model, predictor_type="logit_classifier", num_members=5)

# %%
# Training
# --------
#
# Train each member with the ELBO objective: cross-entropy on the logits plus a
# KL penalty on the variational posterior, mirroring the benchmark recipe.

criterion = ELBOLoss(kl_penalty=1e-5)

for member in credal_model:
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    for _epoch in range(2):
        for inputs, targets in dataloader:
            opt.zero_grad()
            logits = member(inputs)
            kl = collect_kl_divergence(member)
            loss = criterion(logits, targets, kl)
            loss.backward()
            opt.step()
    member.eval()

# %%
# Credal Set Visualization
# ------------------------

rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],
    [0.0, 0.0],
    [0.0, 8.0],
])
credal_sets = rep.predict(X_test)

plot = plot_credal_set(
    credal_sets,
    title="Credal BNN",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD Point", "Near Class 1"],
    show=True,
)
