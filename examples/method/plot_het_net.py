"""====================
HET-Net on Two Moons
====================
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from probly.method.het_net import het_net
from probly.predictor import LogitClassifier
from probly.representer import representer

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

# %%
# Prepare the Two Moons dataset with heteroscedastic noise



X, y = make_moons(n_samples=500, noise=0.0, random_state=0)
rng = np.random.default_rng(0)

noise_scale = np.zeros_like(X[:, 0])
noise_scale[y == 0] = 0.05 + 1.0 * np.clip(-X[y == 0, 0], 0, None)
noise_scale[y == 1] = 0.05 + 1.0 * np.clip(X[y == 1, 0] - 1.5, 0, None)

X += rng.normal(scale=np.expand_dims(noise_scale, 1), size=X.shape)

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Define a sequential model and wrap it with HET-Net
#

base_model = SequentialModel()

het_net_model = het_net(
    base_model,
    predictor_type="logit_classifier"
)
opt = torch.optim.Adam(het_net_model.parameters(), lr=1e-3)

# %%
# Train the HET-Net model
het_net_model.train()

for epoch in range(500):
    out = het_net_model(X_tensor)
    logits = out[0] if isinstance(out, tuple) else out
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# Evaluate Epistemic Uncertainty over a 2D Grid

het_net_model.eval()
rep = representer(het_net_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="HET-Net Predictive Uncertainty")
plot.show()
