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
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Create the DARE ensemble

base_model = MLPClassifier()

dare_model = dare(
    base_model,
    num_members=3,
    reset_params=True,
    predictor_type="logit_classifier",
)

# %%
# Train each member with the DARE anti-regularization term

dare_model.train()
threshold = 0.4
for member in dare_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        out = member(X_tensor)
        loss = nn.functional.cross_entropy(out, y_tensor)
        reg = dare_regularizer(member, device="cpu", loss=loss.detach(), threshold=threshold)
        total = loss - reg

        opt.zero_grad()
        total.backward()
        opt.step()

# %%
# Evaluate predictive uncertainty

dare_model.eval()
rep = IterableSampler(dare_model)

plot = plot_example_uncertainty(X, y, rep, title="DARE Predictive Uncertainty")
plot.show()
