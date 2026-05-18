"""
Evidential Deep Learning on Two Moons
=======================================
"""


from __future__ import annotations

from sklearn.datasets import make_moons
import torch

from probly.representer import representer
from probly.method.evidential import evidential_classification
from probly.train.evidential.torch import evidential_log_loss

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2.Create Evidential Model

base_model = MLPClassifier()

evidential_model = evidential_classification(base_model)

# %%
# 3. Train

opt = torch.optim.Adam(evidential_model.parameters(), lr=1e-3)

evidential_model.train()
for epoch in range(200):
    out = evidential_model(X_tensor)
    loss = evidential_log_loss(out, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

evidential_model.eval()
rep = representer(evidential_model, num_samples=200)

plot = plot_example_uncertainty(X, y, rep, title="Evidential Classification Predictive Uncertainty", vmin = None, vmax = None)
plot.show()
