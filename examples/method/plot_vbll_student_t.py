"""==========================================================
Student-t Variational Bayesian Last Layers on Two Moons
==========================================================

The Student-t variant of Variational Bayesian Last Layers
(:cite:`harrisonVariationalBayesian2024`) extends the discriminative VBLL
classifier by also *inferring* the logit-noise variance instead of fixing it.  A
Gamma variational posterior is placed on the per-class noise precision; once that
uncertain variance is marginalized out, the Gaussian over logits becomes a
**Student-t** distribution with heavier tails, letting the classifier adapt the
noise scale to the data rather than relying on a hand-set value.
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
# Two Moons with a fair amount of observation noise, so inferring the noise scale
# actually matters.

X, y = make_moons(n_samples=500, noise=0.2, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# %%
# Model
# -----
#
# ``vbll(..., variant="student_t")`` replaces the backbone's linear head with a
# ``TVBLLLayer``: a Gaussian posterior over the weights together with a Gamma
# posterior over the per-class noise precision.

vbll_model = vbll(SequentialModel(), variant="student_t", parameterization="dense")

# %%
# Training
# --------
#
# The Student-t variant is trained with the reduced Knowles-Minka softmax bound
# from :cite:`harrisonVariationalBayesian2024`; ``vbll_loss`` dispatches to it
# based on the layer type.  As with the other VBLL layers, the loss needs the
# features feeding the layer, which we capture with a forward pre-hook.

vbll_layer = find_vbll_layer(vbll_model)

captured_features: dict[str, torch.Tensor] = {}
vbll_layer.register_forward_pre_hook(lambda _module, inputs: captured_features.update(features=inputs[0]))


def train_student_t_vbll(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    kl_weight = 1.0 / X.shape[0]
    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        model(X)  # populates captured_features via the pre-hook
        loss = vbll_loss(vbll_layer, captured_features["features"], y, kl_weight)
        loss.backward()
        opt.step()


train_student_t_vbll(vbll_model, X_tensor, y_tensor)

# %%
# Inferred Noise
# --------------
#
# Unlike the standard discriminative VBLL, the Student-t variant learns the noise
# scale.  The expected per-class noise variance is ``rate / (dof - 1)`` of the
# fitted Gamma posterior.

with torch.no_grad():
    expected_noise_var = torch.exp(vbll_layer.noise_log_rate) / (torch.exp(vbll_layer.noise_log_dof) - 1.0)
print("Inferred per-class noise variance:", expected_noise_var.tolist())

# %%
# Uncertainty Evaluation
# ----------------------
#
# The representer samples logits from the (Student-t informed) predictive Gaussian
# and softmaxes them into a categorical sample.  The total uncertainty is high
# along the decision boundary, where the wide noisy band makes the classes ambiguous.

vbll_model.eval()
rep = representer(vbll_model, num_samples=800)

plot = plot_example_uncertainty(
    X,
    y,
    rep,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    title="Student-t VBLL Predictive Uncertainty",
    notion="total",
)
plot.show()
