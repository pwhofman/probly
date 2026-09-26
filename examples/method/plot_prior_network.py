"""============================
Prior Network on Two Moons
============================

A Prior Network parameterizes a Dirichlet distribution over class
probabilities, with concentration parameters ``alpha = exp(logits)``. What
distinguishes it from evidential classification is not so much the model as
the training signal. Rather than learning from in-distribution data alone, a
Prior Network is trained with explicit out-of-distribution (OOD) data:
in-distribution inputs are pushed towards a sharp Dirichlet on the true class,
and OOD inputs towards a flat one. The model is thereby encouraged to separate
aleatoric uncertainty, which arises where the classes overlap, from epistemic
uncertainty, which arises far from the training data.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch import nn
from torch.distributions import Dirichlet, kl_divergence
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from probly.losses.torch import make_ood_target_alpha
from probly.method.prior_network import prior_network
from probly.representer import representer

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
# OOD Data
# --------
#
# Since the method requires OOD inputs during training, these are generated
# synthetically. Points are sampled uniformly from the plotting region, and
# only those at a distance of at least 0.5 from every training point are kept,
# so that no OOD target is placed on the moons.

ood_candidates = torch.rand(5000, 2) * 6 - 3
min_dist = torch.cdist(ood_candidates, X_tensor).min(dim=1).values
X_ood = ood_candidates[min_dist > 0.5]

# %%
# Model
# -----

base_model = MLPClassifier()
pn_model = prior_network(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# The loss mirrors ``pn_loss`` from ``probly.losses.torch``, specialized to
# two classes. It sums the KL divergences between target and predicted
# Dirichlet distributions on in-distribution and OOD inputs, and adds a small
# negative log-likelihood term on the mean class probabilities, which
# stabilizes classification. In-distribution targets have concentration 10 on
# the true class and 1 on the other. The OOD target is the flat Dirichlet
# ``Dir(1, 1)``, as in the original paper. Its low total concentration yields
# a large mutual information, that is, a large epistemic uncertainty, away
# from the data.


def prior_network_loss(x_in: torch.Tensor, y_in: torch.Tensor, x_ood: torch.Tensor) -> torch.Tensor:
    alpha_in = pn_model(x_in)
    target_in = F.one_hot(y_in, 2).float() * 9 + 1
    kl_in = kl_divergence(Dirichlet(target_in), Dirichlet(alpha_in)).mean()

    probs_in = alpha_in / alpha_in.sum(dim=-1, keepdim=True)
    nll = F.nll_loss(torch.log(probs_in + 1e-8), y_in)

    alpha_ood = pn_model(x_ood)
    target_ood = make_ood_target_alpha(x_ood.size(0), num_classes=2, alpha0=2)
    kl_ood = kl_divergence(Dirichlet(target_ood), Dirichlet(alpha_ood)).mean()

    return kl_in + kl_ood + 0.1 * nll


opt = torch.optim.Adam(pn_model.parameters(), lr=1e-3)
grad_clip_norm = 0.5

pn_model.train()
for _epoch in range(200):
    for inputs, targets in dataloader:
        opt.zero_grad()
        ood_idx = torch.randint(0, X_ood.size(0), (inputs.size(0),))
        loss = prior_network_loss(inputs, targets, X_ood[ood_idx])
        loss.backward()
        nn.utils.clip_grad_norm_(pn_model.parameters(), grad_clip_norm)
        opt.step()

# %%
# Evaluation
# ----------
#
# Away from the moons, where the model was trained to predict a flat
# Dirichlet, the epistemic uncertainty is high.

pn_model.eval()
rep = representer(pn_model)

plot = plot_example_uncertainty(X, y, rep, title="Prior Network Epistemic Uncertainty", notion="epistemic")
plot.show()

# %%
# Aleatoric uncertainty, in contrast, concentrates where the two classes are
# hard to tell apart, i.e., near the boundary between the moons.

plot = plot_example_uncertainty(X, y, rep, title="Prior Network Aleatoric Uncertainty", notion="aleatoric")
plot.show()
