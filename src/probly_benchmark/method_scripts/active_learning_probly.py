"""Active learning benchmark using probly's representer + quantifier pipeline.

Demonstrates the probly interface for pool-based active learning:

    active_learning_loop(representer, quantifier_fn, x_train, y_train, x_test, y_test)

The loop builds an internal estimator from the representer (detecting the
predictor type to know how to train) and uses the quantifier for uncertainty
scoring.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from probly.evaluation.active_learning import active_learning_loop  # ty: ignore[unresolved-import]
from probly.method.credal_ensembling import credal_ensembling
from probly.quantification.measure.distribution import entropy_of_expected_value, mutual_information
from probly.representer import representer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple MLP for tabular data
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """Fully-connected network for tabular classification.

    Args:
        n_features: Number of input features.
        n_hidden: Width of each hidden layer.
        n_classes: Number of output classes.
    """

    def __init__(self, n_features: int, n_hidden: int = 64, n_classes: int = 2) -> None:
        """Initialize the network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape ``(batch, n_features)``.

        Returns:
            Logit tensor of shape ``(batch, n_classes)``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X, y = make_classification(n_samples=300, n_features=10, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

N_FEATURES = x_train.shape[1]
N_CLASSES = len(np.unique(y_train))

# ---------------------------------------------------------------------------
# Run: representer + quantifier interface
# ---------------------------------------------------------------------------

strategies = {
    "credal_entropy_of_expected": entropy_of_expected_value,
    "credal_mutual_information": mutual_information,
}

for name, quantifier_fn in strategies.items():
    mlp = SimpleMLP(n_features=N_FEATURES, n_hidden=64, n_classes=N_CLASSES)
    cep = credal_ensembling(mlp, num_members=5, predictor_type="logit_classifier")
    rep = representer(cep, alpha=0.0, distance="euclidean")

    _, _, sc, nauc = active_learning_loop(
        rep,
        quantifier_fn,
        x_train,
        y_train,
        x_test,
        y_test,
        metric="accuracy",
        pool_size=5,
        n_iterations=10,
        n_epochs=20,
        lr=1e-3,
        seed=0,
    )
    logger.info("strategy=%-28s  final_acc=%.4f  nauc=%.4f", name, sc[-1], nauc)
