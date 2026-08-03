"""===============================
Credal DRO Output Visualization
===============================

This example creates an ensemble of standard neural networks on a 3-class
classification problem using ``credal_dro`` and visualizes the predicted
probability intervals (a Probability Intervals Credal Set) for a few test
points using a ternary simplex plot.

Unlike ``credal_wrapper``, whose members differ only by random initialization,
each credal DRO member is trained with distributionally robust optimization at
its own robustness level: member ``i`` backpropagates only the worst
``deltas[i]`` fraction of the losses in each batch
(:cite:`wangLearningCredalEnsembles2026`).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

from probly.method.credal_dro import credal_dro, credal_dro_deltas
from probly.plot.credal import plot_credal_set
from probly.representer import representer
from probly.train.credal.torch import cvar_ce_loss

from examples.utils.model import MLPClassifier

np.random.seed(42)
torch.manual_seed(42)

# %%
# Setup
# -----

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
# Wrap a base classifier with ``credal_dro``: the ensemble structure and the
# probability-interval representer are the same as for ``credal_wrapper``.

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)
credal_model = credal_dro(
    base_model,
    predictor_type="logit_classifier",
    num_members=5,
)

# %%
# Training
# --------
#
# Train each member with the CVaR cross-entropy at its own level: the levels
# interpolate uniformly between the global level ``delta_g`` and 1, so the last
# member is a plain ERM model while earlier members focus on ever-smaller
# fractions of the hardest samples.

deltas = credal_dro_deltas(delta_g=0.5, num_members=5)

for member, delta in zip(credal_model, deltas, strict=True):
    member.train()
    opt = torch.optim.Adam(member.parameters(), lr=1e-2)
    for _epoch in range(1):
        for inputs, targets in dataloader:
            opt.zero_grad()
            logits = member(inputs)
            loss = cvar_ce_loss(logits, targets, delta=delta)
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
    title="Credal DRO Predictions (3-Class)",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD Point", "Near Class 1"],
    show=True,
)
