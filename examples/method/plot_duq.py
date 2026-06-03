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
import torch.nn.functional as F

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
# Train the model with cross-entropy plus a gradient penalty.
# The penalty enforces Lipschitz continuity by keeping gradient norms
# close to 1, ensuring the uncertainty estimates remain reliable.

opt = torch.optim.Adam(duq_model.parameters(), lr=1e-3)
gradient_penalty = 0.5

duq_model.train()
for epoch in range(300):
    num_classes = 2
    targets_onehot = F.one_hot(y_tensor, num_classes).float()

    gradient_penalty = 0.5
    X_tensor.requires_grad_(True)

    kernel_values = duq_model(X_tensor)
    loss = F.binary_cross_entropy(kernel_values, targets_onehot, reduction="mean")

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

plot = plot_example_uncertainty(X, y, rep, title="DUQ Predictive Uncertainty")
plot.show()
