from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.dropout import dropout

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

base_model = MLPClassifier()

dropout_model = dropout(base_model, p=0.25)

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)

dropout_model.train()
for epoch in range(300):
    out = dropout_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()

dropout_model.eval()
rep = representer(dropout_model, num_samples=500)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="Dropout Predictive Uncertainty")
plot.show()
