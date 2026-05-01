"""Torch implementation of Direct Epistemic Uncertainty Prediction (DEUP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import (
    DEUPPredictor,
    DEUPRepresentation,
    create_deup_representation,
    deup_generator,
)


@create_deup_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDEUPRepresentation(DEUPRepresentation, TorchAxisProtected):
    """DEUP representation backed by torch tensors.

    Args:
        softmax: Softmax probabilities of the base classifier,
            shape ``(batch, num_classes)``.
        error_score: Predicted per-sample expected cross-entropy from the
            error head, shape ``(batch,)``.
    """

    softmax: TorchCategoricalDistribution
    error_score: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0}


HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("DEUP_HEAD_MODULE", default=None)


@singledispatch_traverser
def torch_deup_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@torch_deup_traverser.register(nn.Linear)
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the last Linear (hit first under TRAVERSE_REVERSED) with Identity.

    The original layer is saved in ``HEAD_MODULE`` so its shape can be used
    to build the classification and error heads.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    return obj, state


class ErrorPredictionHead(nn.Module):
    r"""MLP head for DEUP that predicts the per-sample cross-entropy error.

    The head maps encoder features :math:`z \in \mathbb{R}^d` to a scalar
    :math:`\hat{e}(x) \in \mathbb{R}`, trained with MSE against the actual
    per-sample cross-entropy of the frozen main model on held-out data.  At
    inference time, :math:`\hat{e}(x)` serves as the total uncertainty score.

    Architecture: ``feature_dim -> (hidden_size x n_hidden_layers) -> 1``
    with ReLU activations between hidden layers.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        r"""Initialize the error prediction head.

        Args:
            feature_dim: Dimensionality of the encoder feature vectors.
            hidden_size: Width of each hidden layer.
            n_hidden_layers: Number of hidden layers (minimum 1).
        """
        super().__init__()
        n_hidden_layers = max(1, n_hidden_layers)
        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-sample cross-entropy from encoder features.

        Args:
            features: Encoder feature vectors of shape ``(batch, feature_dim)``.

        Returns:
            Predicted per-sample cross-entropy of shape ``(batch,)``.
        """
        return self.net(features).squeeze(-1)


@deup_generator.register(nn.Module)
class TorchDEUPPredictor(nn.Module, DEUPPredictor[[torch.Tensor], TorchDEUPRepresentation]):
    """Torch implementation of a DEUP predictor.

    The traversal strips the last ``nn.Linear`` (the classification head) and
    replaces it with ``nn.Identity()``, turning the backbone into a pure
    feature encoder.  The original head is stored as ``classification_head``
    for phase-1 cross-entropy training.  A fresh :class:`ErrorPredictionHead`
    is attached as ``error_head`` and trained in a separate phase on held-out
    per-sample losses.

    **Phase 1** trains ``encoder`` and ``classification_head`` with standard
    cross-entropy (identical to a plain classifier).

    **Phase 2** freezes ``encoder`` and ``classification_head`` and trains
    ``error_head`` on ``(features, CE_loss)`` pairs from the validation set,
    minimising MSE between the predicted error and the actual per-sample
    cross-entropy of the frozen model.
    """

    encoder: nn.Module
    classification_head: nn.Linear
    error_head: ErrorPredictionHead

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 256,
        n_hidden_layers: int = 2,
    ) -> None:
        """Build the three-component DEUP predictor from a base model.

        Args:
            model: Base classification model whose last ``nn.Linear`` defines
                the feature dimension and number of classes.
            hidden_size: Width of each hidden layer in the error head.
            n_hidden_layers: Number of hidden layers in the error head.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(torch_deup_traverser, nn_traverser=nn_traverser),
            init={TRAVERSE_REVERSED: True, HEAD_MODULE: None},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty: ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        self.encoder = encoder
        self.classification_head = head
        self.error_head = ErrorPredictionHead(
            feature_dim=head.in_features,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, classify, and predict per-sample error.

        Args:
            x: Input tensor.

        Returns:
            A 2-tuple of ``(logits, error_score)`` where ``logits`` has shape
            ``(batch, num_classes)`` and ``error_score`` has shape ``(batch,)``.
        """
        features = self.encoder(x)
        logits = self.classification_head(features)
        error_score = self.error_head(features)
        return logits, error_score

    def predict_representation(self, x: torch.Tensor) -> TorchDEUPRepresentation:
        """Predict the DEUP representation (softmax + error score).

        Args:
            x: Input tensor.

        Returns:
            A :class:`TorchDEUPRepresentation` holding the softmax distribution
            and the predicted per-sample cross-entropy.
        """
        logits, error_score = self.forward(x)
        return TorchDEUPRepresentation(
            softmax=TorchCategoricalDistribution(torch.softmax(logits, dim=-1)),
            error_score=error_score,
        )
