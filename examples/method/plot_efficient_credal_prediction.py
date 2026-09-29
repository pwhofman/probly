"""==========================================
Efficient Credal Prediction Visualization
==========================================

This example turns a single trained classifier into a credal predictor with
``efficient_credal_prediction`` and visualizes the resulting probability
interval credal sets on the simplex.

Ensemble-based credal methods obtain the credal set from several independently
trained models. Efficient credal prediction, in contrast, derives it post hoc
from a single model. Roughly speaking, it asks how far each class logit can be
moved before the model explains the training data noticeably worse. More
specifically, for each class, it determines the largest downward and upward
additive offset of that class's logit under which the relative likelihood of
the training data stays above a threshold ``alpha``. At inference time, each
logit is perturbed by these offsets in turn, and the smallest and largest
resulting class probabilities form the lower and upper bounds of the credal
set.

Since the offsets are computed after training, varying ``alpha`` only requires
recomputing them, not retraining the model.
"""

from __future__ import annotations

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from probly.method.efficient_credal_prediction import (
    compute_efficient_credal_prediction_bounds,
    efficient_credal_prediction,
)
from probly.plot.credal import plot_credal_set
from probly.representer import representer

from examples.utils.model import MLPClassifier

# %%
# Setup
# -----
#
# The data consist of three well-separated Gaussian blobs in two dimensions,
# one per class.

centers = [[-7.0, -4.0], [0.0, 8.0], [7.0, -4.0]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=2.0, random_state=42)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Model
# -----

base_model = MLPClassifier(in_features=2, hidden_features=64, out_features=3)
ecp = efficient_credal_prediction(base_model, predictor_type="logit_classifier")

# %%
# Training
# --------
#
# The wrapper leaves the base network unchanged, so the model is trained like
# any ordinary classifier with the cross-entropy loss. Nothing credal happens
# at this stage.

opt = torch.optim.Adam(ecp.parameters(), lr=1e-2)
ecp.train()
for _epoch in range(50):
    for inputs, targets in dataloader:
        opt.zero_grad()
        loss = F.cross_entropy(ecp(inputs), targets)
        loss.backward()
        opt.step()
ecp.eval()

# %%
# Post-hoc Calibration and Visualization
# --------------------------------------
#
# The offsets are calibrated on the training logits, which are therefore
# collected only once. For each threshold ``alpha``, the per-class offsets are
# computed and stored on the predictor as ``lower`` and ``upper``, while the
# trained network itself is reused unchanged. The threshold controls how much
# worse than the trained model a perturbed model may explain the training
# data. A larger ``alpha`` only admits perturbations under which the training
# data remain nearly as likely, and hence yields smaller offsets and narrower
# credal sets.
#
# The effect of ``alpha`` depends on where a point lies. The credal sets of
# the two points at the cluster centers collapse towards a vertex as ``alpha``
# grows, whereas the point between the clusters keeps a wide credal set. Its
# logits are small, so the same offsets suffice to move its prediction across
# the whole simplex.

with torch.no_grad():
    logits_train = ecp(X_train_tensor)

rep = representer(ecp)
X_test = torch.tensor([
    [-7.0, -4.0],
    [0.0, 0.0],
    [0.0, 8.0],
])

for alpha in (0.2, 0.5, 0.8):
    lower, upper = compute_efficient_credal_prediction_bounds(
        logits_train, y_train_tensor, num_classes=3, alpha=alpha
    )
    ecp.lower, ecp.upper = lower.float(), upper.float()

    with torch.no_grad():
        credal_sets = rep.predict(X_test)

    plot_credal_set(
        credal_sets,
        title=f"Efficient Credal Prediction (alpha={alpha})",
        labels=["Class 0", "Class 1", "Class 2"],
        series_labels=["Near Class 0", "OOD", "Near Class 1"],
        show=True,
    )
