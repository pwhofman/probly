"""
DDU on Two Moons
=================

DDU (Deterministic Uncertainty Quantification) uses spectral normalisation
and feature-space density estimation to produce uncertainty estimates.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import numpy as np

from probly.representer import representer
from probly.method.ddu import ddu

from examples.utils.model import ResFFN
from examples.utils.plotting import plot_example_uncertainty

#%%
#1. Prepare the Two Moons dataset
X_raw, y = make_moons(n_samples=500, noise=0.05, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_tensor = torch.from_numpy(X_scaled).float()
y_tensor = torch.from_numpy(y).long()

#%%
#2. Set up the base model
base_model = ResFFN()

ddu_model = ddu(base_model, sn_coeff=5.0, predictor_type="logit_classifier")

opt = torch.optim.Adam(ddu_model.parameters(), lr=1e-3)

#%%
#3. Train the DDU model
ddu_model.train()
for epoch in range(300):

    out = ddu_model(X_tensor)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "mean", out)
    loss = nn.functional.cross_entropy(logits, y_tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()
#%%
#4. Fitting Density Head
ddu_model.eval()

ddu_model.fit_density_head(X_tensor, y_tensor)

#%%
#5. Evaluate predictive uncertainty
rep = representer(ddu_model)

plot = plot_example_uncertainty(X_scaled, y, rep, title="DDU Predictive Uncertainty", vmin=None, vmax=None)
plot.show()
