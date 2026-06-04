"""====================================================================
Evidential Deep Learning on Two Moons
====================================================================

Evidential Deep Learning replaces the softmax output with a Dirichlet
distribution, learning to predict the distribution over class probabilities
directly.  Uncertainty is high when evidence is spread across many classes
or concentrated on a class the model has not seen before.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


from probly.representer import representer
from probly.method.evidential import evidential_classification
from probly.train.evidential.torch import evidential_mse_loss, evidential_kl_divergence


from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Model
# -----
#

base_model = MLPClassifier()
evidential_model = evidential_classification(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# Train using the evidential log-loss, which combines MSE for the evidence
# and a KL-divergence term to regularize the distribution.
# The KL-weight is annealed over the first few epochs to allow the model
# to learn the evidence before enforcing the prior.

opt = torch.optim.Adam(evidential_model.parameters(), lr=1e-3)
grad_clip_norm = 0.5

kl_weight = 0.5
annealing_epochs = 30

evidential_model.train()
for epoch in range (300):

    if annealing_epochs == 0:
        lambda_t = kl_weight
    else:
        lambda_t = kl_weight * min(1.0, epoch / annealing_epochs)

    for inputs, targets in dataloader:

        opt.zero_grad()

        alpha = evidential_model(inputs)

        loss_val = evidential_mse_loss(alpha, targets) + lambda_t * evidential_kl_divergence(alpha, targets)

        loss_val.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(evidential_model.parameters(), grad_clip_norm)

        opt.step()

# %%
# Evaluation
# ----------

evidential_model.eval()
rep = representer(evidential_model, num_samples=200)

plot = plot_example_uncertainty(X, y, rep,title="Evidential Classification Predictive Uncertainty")
plot.show()
