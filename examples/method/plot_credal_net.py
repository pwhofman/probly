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
from probly.representer import representer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# %%
# 1. Prepare a 3-class dataset
# ----------------------------

# Use a wider equilateral triangle configuration so the classes are further away
centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, X_test_data, y_train, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

# %%
# 2. Define a base deterministic model
# ------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

base_model = SimpleMLP()

# %%
# 3. Train base model and wrap with Credal Net
# --------------------------------------------

# Pre-train the base deterministic model
base_model.train()
opt_base = torch.optim.Adam(base_model.parameters(), lr=0.01)
for _ in range(10):
    opt_base.zero_grad()
    logits = base_model(X_train_tensor)
    loss = nn.functional.cross_entropy(logits, y_train_tensor)
    loss.backward()
    opt_base.step()

# Wrap the pre-trained base model into a credal net, inheriting its weights
credal_model = credal_net(base_model, predictor_type="logit_classifier", use_base_weights=True)

# %%
# 4. Predict and plot the credal sets
# -----------------------------------

rep = representer(credal_model)
X_test = torch.tensor([
    [-7.0, -4.0],  # Near Class 0 (Center of blob 0)
    [0.0, 0.0],    # OOD Point (Equidistant from all classes)
    [0.0, 8.0],    # Near Class 1 (Center of blob 1)
])
credal_sets = rep.predict(X_test)

plot_credal_set(
    credal_sets,
    title="Credal Net",
    labels=["Class 0", "Class 1", "Class 2"],
    series_labels=["Near Class 0", "OOD", "Near Class 1"],
    show=True,
)
