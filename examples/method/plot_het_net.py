"""
HET-Net on Two Moons
=====================
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
import torch.nn.functional as F

from probly.method.het_net import het_net
from probly.predictor import LogitClassifier
from probly.representer import representer

from examples.utils.plotting import plot_example_uncertainty
from examples.utils.model import SequentialModel

# %%
# 1. Prepare the Two Moons dataset with heteroscedastic noise

import numpy as np

X, y = make_moons(n_samples=500, noise=0.0, random_state=0)
rng = np.random.default_rng(0)

# Introduce flexible heteroscedastic noise with the same inverted profile for each moon
noise_scale = np.zeros_like(X[:, 0])
# Class 0 (typically blue): Inverted profile, decreases towards the right (noisy -> less noisy)
noise_scale[y == 0] = 0.1 + 0.3 * np.clip(1.0 - X[y == 0, 0], 0, None)
# Class 1 (typically orange): Inverted profile, decreases towards the right (noisy -> less noisy)
noise_scale[y == 1] = 0.1 + 0.3 * np.clip(2.0 - X[y == 1, 0], 0, None)

X += rng.normal(scale=np.expand_dims(noise_scale, 1), size=X.shape)

X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# 2. Define a sequential model and wrap it with HET-Net
#

base_model = SequentialModel()

# 2. Configure HET-Net parameters
het_net_model = het_net(
    base_model,
    predictor_type="logit_classifier"
)
opt = torch.optim.Adam(het_net_model.parameters(), lr=1e-3)

# %%
# 3. Train the HET-Net model
het_net_model.train()

# 4. Train for an appropriate number of epochs
for epoch in range(500):
    out = het_net_model(X_tensor)
    logits = out[0] if isinstance(out, tuple) else out
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()

# %%
# 4. Evaluate Epistemic Uncertainty over a 2D Grid

het_net_model.eval()
rep = representer(het_net_model, num_samples=800)

plot = plot_example_uncertainty(X, y, rep, title="HET-Net Predictive Uncertainty")
plot.show()
