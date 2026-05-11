from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.ddu import ddu

from examples.utils.model import MiniResNet1D
from examples.utils.plotting_decomp import plot_example_uncertainty_decomp

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = MiniResNet1D(n_classes=2)

ddu_model = ddu(base_model, sn_coeff=1.5)

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)

ddu_model.train()
for epoch in range(300):
    out = ddu_model(X_tensor)
    mean_attr = getattr(out, "mean", None)
    logits = out[0] if isinstance(out, tuple) else (mean_attr if isinstance(mean_attr, torch.Tensor) else out)
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

ddu_model.eval()
rep = representer(ddu_model)

plot = plot_example_uncertainty_decomp(X, X_tensor, y, rep, title="DDU Predictive Uncertainty")
plot.show()
