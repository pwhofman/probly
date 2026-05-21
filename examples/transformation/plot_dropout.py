"""=======================
MC Dropout on Two Moons
=======================

MC Dropout keeps dropout active at inference time, producing stochastic
predictions that can be averaged to estimate predictive uncertainty.
Uncertainty concentrates at the decision boundary between classes.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import dropout

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Wrap the base model with MC Dropout

base_model = MLPClassifier()

dropout_model = dropout(base_model, p=0.25, predictor_type="logit_classifier",)

# %%
# Train

opt = torch.optim.Adam(dropout_model.parameters(), lr=1e-3)

dropout_model.train()
for epoch in range(300):
    out = dropout_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# Evaluate predictive uncertainty

dropout_model.eval()
rep = representer(dropout_model, num_samples=200)

plot = plot_example_uncertainty(X, y, rep, title="Dropout Predictive Uncertainty")
plot.show()
