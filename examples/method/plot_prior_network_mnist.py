"""=========================
Prior Network on MNIST
=========================

A Prior Network parameterizes a Dirichlet distribution over class
probabilities, with concentration parameters ``alpha = exp(logits)``, and is
trained with explicit out-of-distribution (OOD) data: in-distribution digits
are pushed towards a sharp Dirichlet on the true class, and OOD inputs towards
a flat one. Here, uniform noise images serve as OOD training data. This is
admittedly the simplest possible choice, and the example is meant to
illustrate the mechanism rather than a realistic OOD detection setup.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probly.losses.torch import pn_loss
from probly.method.prior_network import prior_network
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.data import load_mnist

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_mnist_uncertainty

# %%
# Setup
# -----

train_loader, test_loader = load_mnist(batch_size=256)

X_test_batches, y_test_batches = zip(*test_loader)
X_test = torch.cat([x.view(-1, 28 * 28) for x in X_test_batches])
y_test = torch.cat(list(y_test_batches))
images_test = (X_test.view(-1, 28, 28) * 255).byte()

# %%
# Model
# -----

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
pn_model = prior_network(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# ``pn_loss`` sums the KL divergences from sharp in-distribution targets and
# flat OOD targets to the predicted Dirichlet distributions, and adds a small
# cross-entropy term. Each batch of flattened training images is paired with
# freshly sampled uniform noise images as OOD inputs. Note that the
# in-distribution targets infer the number of classes from the batch labels.
# The loader therefore uses ``drop_last=True``, which avoids a small final
# batch that might miss a digit.

X_train_batches, y_train_batches = zip(*train_loader)
X_train_flat = torch.cat([x.view(-1, 28 * 28) for x in X_train_batches])
y_train = torch.cat(list(y_train_batches))

flat_dataloader = DataLoader(
    TensorDataset(X_train_flat, y_train),
    batch_size=256,
    shuffle=True,
    drop_last=True,
)

opt = torch.optim.Adam(pn_model.parameters(), lr=1e-3)
grad_clip_norm = 0.5

pn_model.train()
for _epoch in range(5):
    for inputs, targets in flat_dataloader:
        opt.zero_grad()
        x_ood = torch.rand_like(inputs)
        loss = pn_loss(pn_model, inputs, targets, x_ood)
        loss.backward()
        nn.utils.clip_grad_norm_(pn_model.parameters(), grad_clip_norm)
        opt.step()

# %%
# Uncertainty Quantification
# --------------------------
#
# The epistemic uncertainty of each predicted Dirichlet is quantified with
# ``quantify`` and converted from nats to bits.

pn_model.eval()
rep = representer(pn_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
uncertainty = uq.epistemic.detach().numpy() / np.log(2)

# %%
# Predictions
# -----------
#
# Point predictions are the means of the predicted Dirichlet distributions,
# ``alpha / sum(alpha)``.

with torch.no_grad():
    alpha = pn_model(X_test)
    mean_probs = (alpha / alpha.sum(dim=-1, keepdim=True)).numpy()

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (Prior Network)",
)
plot.show()

# %%
# In-Distribution vs. OOD
# -----------------------
#
# The model was trained to predict a flat Dirichlet on noise, and unseen
# uniform noise images accordingly receive a much higher epistemic
# uncertainty than almost all MNIST test digits. Note, however, that these
# images come from the same distribution as the OOD training data. The
# histogram therefore shows that the model has learned the separation it was
# trained for, not that it detects OOD inputs of a different kind.

X_noise = torch.rand(len(X_test), 28 * 28)
with torch.no_grad():
    uncertainty_noise = quantify(rep.represent(X_noise)).epistemic.numpy() / np.log(2)

fig, ax = plt.subplots(figsize=(7, 4))
all_uncertainty = np.concatenate([uncertainty, uncertainty_noise])
bins = np.linspace(all_uncertainty.min(), all_uncertainty.max(), 50)
ax.hist(uncertainty, bins=bins, alpha=0.6, label="MNIST test")
ax.hist(uncertainty_noise, bins=bins, alpha=0.6, label="Uniform noise")
ax.set_xlabel("Epistemic uncertainty [bits]")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
plt.show()
