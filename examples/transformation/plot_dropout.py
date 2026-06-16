"""=======================
MC Dropout on Two Moons
=======================

Keep dropout active at inference and average several stochastic forward passes.
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
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----

base_model = MLPClassifier()

dropout_model = dropout(
    base_model,
    p=0.5,  # zeroing probability (kept active at inference)
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Standard cross-entropy with mini-batches.  Dropout stays active at
# inference time, which is what enables repeated forward passes to produce
# a distribution over predictions.

opt = torch.optim.Adam(dropout_model.parameters(), lr=1e-3)

dropout_model.train()
for epoch in range(300):
    opt.zero_grad()
    out = dropout_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

dropout_model.eval()
rep = representer(dropout_model, num_samples=400, shared_dropout_mask=True) # shared_dropout_mask=True to apply a single shared binary mask over the output instead of one per batch element

plot = plot_example_uncertainty(X, y, rep, title="Dropout Predictive Uncertainty", notion="total")
plot.show()
