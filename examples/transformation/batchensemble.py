from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.batchensemble import batchensemble

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = SequentialModel()


num_members = 3
batchensemble_model = batchensemble(base_model, num_members=num_members)

batchensemble_model.train()
opt = torch.optim.Adam(batchensemble_model.parameters(), lr=1e-3)

# For training BatchEnsemble, we tile the batch `num_members` times (shape: [E*B, ...])
X_tiled = X_tensor.repeat(num_members, 1)
y_tiled = y_tensor.repeat(num_members)

for epoch in range(300):
    opt.zero_grad()
    out = batchensemble_model(X_tiled)
    loss = nn.functional.cross_entropy(out, y_tiled)
    loss.backward()
    opt.step()

batchensemble_model.eval()
# The representer automatically handles tiling the inputs and reshaping the outputs for prediction!
rep = representer(batchensemble_model, num_samples=800)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="BatchEnsemble Predictive Uncertainty")
plot.show()
