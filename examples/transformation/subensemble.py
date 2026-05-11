

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method import subensemble

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = SequentialModel()

subensemble_model = subensemble(
    base_model,
    num_heads=3,
    reset_params=True,
)

subensemble_model.train()
opt = torch.optim.Adam(subensemble_model.parameters(), lr=1e-3)
for epoch in range(300):
    opt.zero_grad()
    total_loss = 0.0
    for member in subensemble_model:
        out = member(X_tensor)
        total_loss = total_loss + nn.functional.cross_entropy(out, y_tensor)

    total_loss.backward()
    opt.step()

subensemble_model.eval()
rep = representer(subensemble_model)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="Sub-Ensemble Predictive Uncertainty")
plot.show()
