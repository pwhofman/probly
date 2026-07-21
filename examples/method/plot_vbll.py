"""=============================================
Variational Bayesian Last Layers on Two Moons
=============================================

Variational Bayesian Last Layers (VBLL, :cite:`harrisonVariationalBayesian2024`)
replace a network's final dense layer with a Bayesian last layer that maintains a
variational posterior over the output weights.  At inference it emits, in closed
form, a Gaussian over the logits whose variance grows as inputs leave the training
distribution, so the predictive captures epistemic uncertainty that increases
smoothly out of distribution.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn

from probly.method.vbll import find_vbll_layer, vbll
from probly.representer import representer
from probly.train.vbll import vbll_loss

from examples.utils.model import SequentialModel
from examples.utils.plotting import plot_example_uncertainty

torch.manual_seed(0)

# %%
# Setup
# -----
#
# Prepare the Two Moons dataset.

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()


# %%
# Model
# -----
#
# ``vbll`` replaces the backbone's linear output head with a ``VBLLLayer`` that
# holds a dense variational posterior over the last-layer weights.

vbll_model = vbll(SequentialModel(), parameterization="dense")

# %%
# Training
# --------
#
# VBLL is trained by maximizing the deterministic ELBO of
# :cite:`harrisonVariationalBayesian2024`: the closed-form double-Jensen softmax
# bound plus the weight-posterior KL term, computed by ``vbll_loss``.  As with the
# other VBLL layers, the loss needs the features feeding the layer, which we capture
# with a forward pre-hook.

vbll_layer = find_vbll_layer(vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))


def train_vbll(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 500) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    kl_weight = 1.0 / X.shape[0]
    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        model(X)  # populates captured_features via the pre-hook
        loss = vbll_loss(vbll_layer, captured_features["features"], y, kl_weight)
        loss.backward()
        opt.step()


train_vbll(vbll_model, X_tensor, y_tensor)

# %%
# Uncertainty Evaluation
# ----------------------
#
# The representer draws logit samples from the predictive Gaussian and softmaxes
# them into a categorical sample, whose total uncertainty grows away from the data.

vbll_model.eval()
rep = representer(vbll_model, num_samples=800)

plot = plot_example_uncertainty(
    X,
    y,
    rep,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    title="VBLL Predictive Uncertainty",
    notion="total",
)
plot.show()
