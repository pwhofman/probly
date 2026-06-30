"""============================
Masksembles on MNIST
============================

Replace a full ensemble with a fixed set of binary masks inserted after each hidden layer of a shared backbone.
During training one mask is drawn per sample (dropout-style); during inference the model runs once per mask.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from probly.predictor import predict
from probly.quantification import quantify
from probly.representer import representer
from probly.transformation.masksembles import masksembles
from probly_benchmark.data import load_mnist

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_mnist_uncertainty

# %%
# Setup
# -----

train_loader, test_loader = load_mnist(batch_size=64)

X_test_batches, y_test_batches = zip(*test_loader)
X_test = torch.cat([x.view(-1, 28 * 28) for x in X_test_batches])
y_test = torch.cat(list(y_test_batches))
images_test = (X_test.view(-1, 28, 28) * 255).byte()

# %%
# Model
# -----
#
# ``n_masks`` binary masks are generated and inserted after each linear layer.
# A larger ``scale`` reduces overlap (and thus correlation) between masks at
# the cost of capacity per masked sub-network.

base_model = MLPClassifier(in_features=28 * 28, hidden_features=256, out_features=10)
n_masks = 4
masksembles_model = masksembles(
    base_model,
    n_masks=n_masks, # The higher, the more similar to MC Dropout
    scale=2.0, # The higher, the more similar to Ensemble
    predictor_type="logit_classifier",
)

print(masksembles_model)

# %%
# Training
# --------
#
# Fine-tune with standard cross-entropy. In training mode each sample in the
# batch is masked independently with a uniformly random mask (dropout-style);
# no manual batch tiling is needed.

masksembles_model.train()
opt = torch.optim.Adam(masksembles_model.parameters(), lr=1e-4)

epochs = 5

masksembles_model.train()
for epoch in range(epochs):
    train_loss, train_correct = 0, 0
    for x_batch, y_batch in train_loader:
        X_flat = x_batch.view(-1, 28 * 28)

        opt.zero_grad()
        out = masksembles_model(X_flat)
        loss = nn.functional.cross_entropy(out, y_batch)
        loss.backward()
        opt.step()

        train_loss += loss.item() * X_flat.size(0)
        train_correct += (out.argmax(1) == y_batch).sum().item()

    # Validation
    masksembles_model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            X_flat = x_batch.view(-1, 28 * 28)
            out = masksembles_model(X_flat)
            loss = nn.functional.cross_entropy(out, y_batch)
            val_loss += loss.item() * X_flat.size(0)
            val_correct += (out.argmax(1) == y_batch).sum().item()

    masksembles_model.train()
    print(
        f"Epoch {epoch+1}/{epochs} "
        f"- Train loss: {train_loss/len(train_loader.dataset):.4f}, "
        f"Train acc: {train_correct/len(train_loader.dataset):.4f}, "
        f"Val loss: {val_loss/len(test_loader.dataset):.4f}, "
        f"Val acc: {val_correct/len(test_loader.dataset):.4f}"
    )
# %%
# Uncertainty Quantification
# --------------------------

masksembles_model.eval()
rep = representer(masksembles_model)

with torch.no_grad():
    representation = rep.represent(X_test)

uq = quantify(representation)
_total = uq.total
uncertainty = (
    _total.detach().numpy() if isinstance(_total, torch.Tensor) else np.asarray(_total)
)
uncertainty = uncertainty / np.log(2)
if uncertainty.ndim > 1:
    uncertainty = uncertainty.sum(axis=-1)

# %%
# Predictions
# -----------
#
# ``predict`` tiles the input by ``n_masks`` internally and returns a
# ``TorchSample`` of shape ``[n_masks, N, num_classes]`` — one slice per mask.
# Softmax converts logits to probabilities; averaging over masks gives the
# mean predictive distribution used for the final class prediction.

with torch.no_grad():
    sample = predict(masksembles_model, X_test)              # TorchSample [n_masks, N, 10]

member_probs = sample.tensor.softmax(-1).numpy()             # [n_masks, N, 10]
mean_probs = member_probs.mean(axis=0)                       # [N, 10]

accuracy = (mean_probs.argmax(-1) == y_test.numpy()).mean() * 100
print(f"Test accuracy: {accuracy:.1f}%")

# %%
# Visualization
# -------------
#
# Plot the five most uncertain test digits with per-member agreement.

plot = plot_mnist_uncertainty(
    images_test,
    y_test,
    uncertainty,
    mean_probs,
    title="Top-5 Most Uncertain Test Predictions (Masksembles)",
)
plot.show()
