"""================
DUQ on Two Moons
================

Deep Uncertainty Quantification (DUQ) replaces the standard softmax output with a
radial basis function (RBF) network that maps feature representations to per-class
centroids.
The epistemic uncertainty can be estimated by measuring the kernel distance
between an input's representation and these learned centroids.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.method.duq import duq

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Wrap the base model with DropConnect

base_model = MLPClassifier()

duq_model = duq(base_model, predictor_type="logit_classifier")

# %%
# Train

opt = torch.optim.Adam(duq_model.parameters(), lr=1e-3)

duq_model.train()
for epoch in range(300):
    out = duq_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# Evaluate predictive uncertainty

duq_model.eval()
rep = representer(duq_model)

plot = plot_example_uncertainty(X, y, rep, title="DUQ Predictive Uncertainty", vmin = None, vmax = None)
plot.show()
