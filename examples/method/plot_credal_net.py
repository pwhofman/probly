"""========================
Credal Net Visualization
========================

This example uses ``credal_net`` to build an interval-arithmetic classifier
and visualize the resulting probability interval credal sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.method.credal_net import credal_net
from probly.plot.credal import plot_credal_set
from probly.predictor import predict_raw
from probly.representer import representer
from probly.train.credal.torch import intersection_probability_ce_loss

from examples.utils.model import MLPClassifier


np.random.seed(42)
torch.manual_seed(42)

# %%
# Setup
# -----

centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, X_test_data, y_train, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

# %%
# Model
# -----
#
# Wrap a base classifier with ``credal_net`` so each weight becomes a learnable
# interval; ``use_base_weights=True`` initializes the interval centers from the
# (untrained) base weights.

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)
prob_model = nn.Sequential(base_model, nn.Softmax(dim=1))
credal_model = credal_net(prob_model, predictor_type="probabilistic_classifier", use_base_weights=True)

# %%
# Training
# --------
#
# Train the wrapped credal net directly with the intersection-probability
# cross-entropy loss (Eq. 14 of :cite:`wangCredalDeepEnsembles2024`), which operates on
# the packed ``(lower, upper)`` interval output produced by ``predict_raw``.

opt = torch.optim.Adam(credal_model.parameters(), lr=1e-2)

credal_model.train()
for _epoch in range(10):
    opt.zero_grad()
    output = predict_raw(credal_model, X_train_tensor)
    loss = intersection_probability_ce_loss(output, y_train_tensor)
    loss.backward()
    opt.step()

# %%
# Credal Set Visualization
# ------------------------

credal_model.eval()
rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],
    [0.0, 0.0],
    [0.0, 8.0],
])
credal_sets = rep.predict(X_test)

plot_credal_set(
    credal_sets,
    title="Credal Net",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD", "Near Class 1"],
    show=True,
)
