"""========================
DropConnect on Two Moons
========================

Mask individual weights instead of activations and average several stochastic forward passes at inference.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import dropconnect

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----

base_model = MLPClassifier()

dropconnect_model = dropconnect(
    base_model,
    p=0.5,  # per-weight masking probability (kept active at inference)
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Standard cross-entropy with mini-batches.  Weight masking is applied at
# every forward pass, both during training and at inference time.

opt = torch.optim.Adam(dropconnect_model.parameters(), lr=1e-3)

dropconnect_model.train()
for epoch in range(300):
    opt.zero_grad()
    out = dropconnect_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

dropconnect_model.eval()
rep = representer(dropconnect_model, num_samples=400)

plot = plot_example_uncertainty(X, y, rep, title="DropConnect Predictive Uncertainty", notion="total")
plot.show()
