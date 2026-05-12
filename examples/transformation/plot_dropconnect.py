"""=======================================
DropConnect on Two Moons
=======================================

DropConnect randomly drops individual weights rather than activations at
inference time, producing stochastic predictions similar to MC Dropout.
Uncertainty concentrates at the decision boundary between classes.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.representer import representer
from probly.transformation import dropconnect

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# 1. Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Wrap the base model with DropConnect

base_model = MLPClassifier()

dropconnect_model = dropconnect(base_model, p=0.25)

# %%
# 3. Train

opt = torch.optim.Adam(dropconnect_model.parameters(), lr=1e-3)

dropconnect_model.train()
for epoch in range(300):
    out = dropconnect_model(X_tensor)
    loss = nn.functional.cross_entropy(out, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate predictive uncertainty

dropconnect_model.eval()
rep = representer(dropconnect_model, num_samples=400)

plot = plot_example_uncertainty(X, X_tensor, y, rep, title="DropConnect Predictive Uncertainty")
plot.show()
