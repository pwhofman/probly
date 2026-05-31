"""==========================
Deep Ensemble on Two Moons
==========================

Train several independent copies of the same network from different initializations and average their predictions.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import ensemble

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

ensemble_model = ensemble(
    base_model,
    num_members=3,
    reset_params=True,  # fresh init per member maximizes diversity
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Train each member independently with standard cross-entropy.
# The uncertainty signal comes from disagreement between members,
# not from the loss itself.

ensemble_model.train()
for member in ensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        opt.zero_grad()
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)
        loss.backward()
        opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

ensemble_model.eval()
rep = representer(ensemble_model)

plot = plot_example_uncertainty(X, y, rep, title="Ensemble Predictive Uncertainty")
plot.show()
