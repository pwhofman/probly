from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.ensemble import ensemble

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = MLPClassifier()

ensemble_model = ensemble(
    base_model,
    num_members=3,
    reset_params=True,
)

ensemble_model.train()
for member in ensemble_model:
    opt = torch.optim.Adam(member.parameters(), lr=1e-3)
    for epoch in range(250):
        out = member(X_tensor)
        mean_attr = getattr(out, "mean", None)
        logits = out[0] if isinstance(out, tuple) else (mean_attr if isinstance(mean_attr, torch.Tensor) else out)
        loss = nn.functional.cross_entropy(logits, y_tensor)

        opt.zero_grad()
        loss.backward()
        opt.step()

ensemble_model.eval()
rep = representer(ensemble_model)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="Ensemble Predictive Uncertainty")
plot.show()
