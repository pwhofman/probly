from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.dropconnect import dropconnect

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = MLPClassifier()

dropconnect_model = dropconnect(base_model, p=0.25)

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)

dropconnect_model.train()
for epoch in range(300):
    out = dropconnect_model(X_tensor)
    mean_attr = getattr(out, "mean", None)
    logits = out[0] if isinstance(out, tuple) else (mean_attr if isinstance(mean_attr, torch.Tensor) else out)
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

dropconnect_model.eval()
rep = representer(dropconnect_model, num_samples=800)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="DropConnect Predictive Uncertainty")
plot.show()
