"""================
DUQ on Two Moons
================

Deep Uncertainty Quantification (DUQ) replaces the softmax head with a radial
basis function (RBF) network that maps feature representations to per-class
centroids.  Uncertainty is estimated from the kernel distances between an
input's representation and the learned centroids.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
import torch.nn.functional as F
from torch import nn

from probly.representer import representer
from probly.method.duq import duq

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty


# %%
# Setup
# -----

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----

base_model = MLPClassifier()
duq_model = duq(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# DUQ uses binary cross-entropy on the kernel outputs together with a gradient
# penalty that enforces a bi-Lipschitz constraint on the feature map.

opt = torch.optim.Adam(duq_model.parameters(), lr=1e-3)
criterion = nn.BCELoss(reduction = "mean")

gradient_penalty = 0.5
num_classes = 2

duq_model.train()
for epoch in range(300):
    X_tensor.requires_grad_(True)
    targets_onehot = F.one_hot(y_tensor, num_classes).float()

    kernel_values = duq_model(X_tensor)
    loss = criterion(kernel_values, targets_onehot)

    gradients = torch.autograd.grad(
        outputs=kernel_values,
        inputs=X_tensor,
        grad_outputs=torch.ones_like(kernel_values),
        create_graph=True,
        retain_graph=True,
    )[0]
    flat_gradients = gradients.flatten(start_dim=1)
    grad_norm = flat_gradients.norm(2, dim=1)
    duq_penalty = ((grad_norm - 1.0) ** 2).mean()

    total_loss = loss + gradient_penalty * duq_penalty

    opt.zero_grad()
    total_loss.backward()
    opt.step()

# %%
# Uncertainty Evaluation
# ----------------------

duq_model.eval()
rep = representer(duq_model)

plot = plot_example_uncertainty(X, y, rep, title="DUQ Predictive Uncertainty", notion="total")
plot.show()
