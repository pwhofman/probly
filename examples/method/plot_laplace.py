"""==================================
Laplace Approximation on Two Moons
==================================

The Laplace Approximation is a post-hoc method that turns a deterministically
trained neural network into a Bayesian Neural Network by approximating the
posterior over weights with a Gaussian distribution. Uncertainty concentrates
along the decision boundary.
"""

from __future__ import annotations

from laplace import Laplace
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.representer import representer

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
fit_loader = DataLoader(dataset, batch_size=32)

# %%
# 2. Train the base deterministic model

base_model = MLPClassifier()

opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)

base_model.train()
for epoch in range(300):
    out = base_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 3. Fit the Laplace approximation
#
# We fit the approximation on the last layer of the trained model
# using a Kronecker-factored (KFAC) Hessian structure.

base_model.eval()

laplace_model = Laplace(
    base_model,
    "classification",
    subset_of_weights="last_layer",
    hessian_structure="kron"
)
laplace_model.fit(fit_loader)

# %%
# 4. Evaluate predictive uncertainty

rep = representer(laplace_model, num_samples=200)

plot = plot_example_uncertainty(X, y, rep, title="Laplace Predictive Uncertainty")
plot.show()
