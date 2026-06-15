"""=================
DARE on Two Moons
=================

DARE (Deep Anti-Regularized Ensembles) adds a per-member anti-regularization
term, active once the task loss drops below a threshold, that pushes each
member's weights to larger magnitudes. Preventing weight collapse preserves
the diversity introduced by different initializations, improving ensemble
out-of-distribution detection.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import IterableSampler
from probly.method.dare import dare
from probly.train.dare.torch import dare_regularizer

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
#
# DARE wraps an ensemble of independent members. Each member is trained with
# an anti-regularization term that fires when the per-batch cross-entropy drops
# below `threshold`, pushing weights to larger norms and preserving diversity.

base_model = MLPClassifier()

dare_model = dare(
    base_model,
    num_members=3,
    reset_params=True,
    predictor_type="logit_classifier",
)

# %%
# Training
# --------
#
# Train each member with cross-entropy minus the DARE anti-regularization term.
# The anti-regularizer only activates once the batch loss falls below threshold.

dare_model.train()
threshold = 0.4
for member in dare_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        opt.zero_grad()
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)
        reg = dare_regularizer(member, device="cpu", loss=loss.detach(), threshold=threshold)
        total = loss - reg
        total.backward()
        opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

dare_model.eval()
rep = IterableSampler(dare_model)

plot = plot_example_uncertainty(X, y, rep, title="DARE Predictive Uncertainty", notion="total")
plot.show()
