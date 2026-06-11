"""=================
Het-Net on MNIST
=================

Het-Net adds a learnable heteroscedastic noise head to the base classifier.
The representer draws stochastic samples from this noise to estimate the
predictive distribution, capturing per-sample aleatoric uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from probly.layers.torch import HeteroscedasticLayer
from probly.method.het_net import het_net
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
#
# Het-Net augments a standard classifier with a per-sample noise head.
# The noise head is co-trained and learns where the model is uncertain
# due to irreducible label noise.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
het_net_model = het_net(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# Setting ``training_samples = S`` on every ``HeteroscedasticLayer`` makes the
# head draw S noise samples per input in a single vectorized forward pass and
# return the log of the softmax-averaged probabilities, optimized with NLL.

opt = torch.optim.Adam(het_net_model.parameters(), lr=1e-3)
training_samples = 4

het_layers = [m for m in het_net_model.modules() if isinstance(m, HeteroscedasticLayer)]
for layer in het_layers:
    layer.training_samples = training_samples

het_net_model.train()
try:
    for _epoch in range(5):
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            X_flat = X_batch.view(-1, 28 * 28)
            opt.zero_grad()
            log_probs = het_net_model(X_flat)
            loss = F.nll_loss(log_probs, y_batch)
            loss.backward()
            opt.step()
            correct += (log_probs.detach().argmax(-1) == y_batch).sum().item()
            total += len(y_batch)
        if correct / total >= 0.97:
            break
finally:
    for layer in het_layers:
        layer.training_samples = 1

# %%
# Uncertainty Quantification
# --------------------------
#
# The representer stochastically samples the noise head to build a
# second-order distribution over the output.

het_net_model.eval()
rep = representer(het_net_model, num_samples=800)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_unc = uq.total if hasattr(uq, "total") else (uq.epistemic if hasattr(uq, "epistemic") else uq.aleatoric)
uncertainty = _unc.detach().numpy() if isinstance(_unc, torch.Tensor) else np.asarray(_unc)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------

with torch.no_grad():
    out = het_net_model(X_test)
    logits = out[0] if isinstance(out, tuple) else out
    mean_probs = logits.softmax(-1).numpy()

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
    title="Top-5 Most Uncertain Test Predictions (Het-Net)",
)
plot.show()
