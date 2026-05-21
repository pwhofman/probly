"""====================================================================
Evidential Deep Learning on Two Moons
====================================================================

Evidential Deep Learning replaces the standard softmax output with
a Dirichlet distribution over a simplex. The output is an evidence
vector, the method therefore directly learns evidence for each class to
directly predict the distribution.
"""

from __future__ import annotations

from sklearn.datasets import make_moons
import torch
from torch.utils.data import TensorDataset, DataLoader

from probly.representer import representer
from probly.method.evidential import evidential_classification
from probly.train.evidential.torch import evidential_log_loss
from probly.train.evidential.torch import unified_evidential_train

from examples.utils.model import MLPClassifier
from examples.utils.plotting import plot_example_uncertainty

# %%
# Prepare the Two Moons dataset

X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()


dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Wrap the base model with DropConnect

base_model = MLPClassifier()
evidential_model = evidential_classification(base_model, predictor_type="logit_classifier")

# %%
# Train using unified_evidential_train

unified_evidential_train(
    mode="EDL",
    model=evidential_model,
    dataloader=dataloader,
    loss_fn=evidential_log_loss,
    oodloader=None,
    class_count=None,
    epochs=300,
    lr=1e-3,
    device="cpu"
)

# %%
# Evaluate predictive uncertainty

evidential_model.eval()
rep = representer(evidential_model, num_samples=200)

plot = plot_example_uncertainty(
    X, y, rep,
    title="Evidential Classification Predictive Uncertainty",
    vmin=None,
    vmax=None
)
plot.show()
