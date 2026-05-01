"""Torch implementation of the DUQ transformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import (
    DUQPredictor,
    DUQRepresentation,
    create_duq_representation,
    duq_generator,
    duq_uncertainty,
)

if TYPE_CHECKING:
    from torch import Tensor


@create_duq_representation.register(torch.Tensor)
@dataclass(frozen=True, slots=True)
class TorchDUQRepresentation(DUQRepresentation, TorchAxisProtected):
    """DUQ representation backed by a torch tensor."""

    kernel_values: Tensor
    protected_axes: ClassVar[dict[str, int]] = {"kernel_values": 1}


@duq_uncertainty.register(torch.Tensor)
def torch_duq_uncertainty(kernel_values: torch.Tensor) -> torch.Tensor:
    r"""Per-sample DUQ uncertainty :math:`1 - \max_c K_c(x)` for torch tensors."""
    return 1.0 - kernel_values.max(dim=-1).values


HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("RBF_CENTROID_HEAD_MODULE", default=None)


@singledispatch_traverser
def torch_duq_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@torch_duq_traverser.register
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the classification head (last Linear, hit first under TRAVERSE_REVERSED) with Identity."""
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    return obj, state


class RBFCentroidHead(nn.Module):
    r"""RBF centroid head for DUQ :cite:`vanAmersfoortDUQ2020`.

    Maintains a learnable per-class projection
    :math:`W \in \mathbb{R}^{n \times C \times d}` and EMA-tracked class
    centroid statistics: ``centroid_counts`` :math:`N \in \mathbb{R}^C` and
    ``centroids_sum`` :math:`m \in \mathbb{R}^{n \times C}` with
    :math:`e_c = m_c / N_c`. ``forward`` returns the per-class kernel values
    :math:`K_c(x) = \exp\left(-\|W_c f - e_c\|^2 / (2 n \sigma^2)\right)`,
    shape ``(batch, num_classes)``.

    Buffers:
        centroid_counts: EMA-tracked class counts, shape ``(num_classes,)``.
        centroids_sum: EMA-tracked sums of per-class embeddings,
            shape ``(centroid_size, num_classes)``.
    """

    centroid_counts: torch.Tensor
    centroids_sum: torch.Tensor

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        centroid_size: int = 256,
        length_scale: float = 0.1,
        gamma: float = 0.999,
    ) -> None:
        r"""Initialize the centroid head.

        Args:
            feature_dim: Dimensionality of the encoder feature vectors.
            num_classes: Number of classes.
            centroid_size: Embedding dimension :math:`n`.
            length_scale: RBF kernel length scale :math:`\sigma`.
            gamma: Exponential moving-average decay for the centroid statistics.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.centroid_size = centroid_size
        self.length_scale = length_scale
        self.gamma = gamma

        weight = torch.empty(centroid_size, num_classes, feature_dim)
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight)

        # Initial counts and sums follow the reference implementation:
        # counts start at a small positive value and embeddings are seeded with
        # small Gaussian noise so that early centroids are well-defined.
        self.register_buffer("centroid_counts", torch.full((num_classes,), 13.0))
        sums = torch.empty(centroid_size, num_classes).normal_(0.0, 0.05) * 13.0
        self.register_buffer("centroids_sum", sums)

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to the per-class embedding space.

        Args:
            features: Feature vectors of shape ``(batch, feature_dim)``.

        Returns:
            Per-class embeddings of shape ``(batch, centroid_size, num_classes)``.
        """
        return torch.einsum("ncd,bd->bnc", self.weight, features)

    @property
    def centroids(self) -> torch.Tensor:
        """Class centroids :math:`e_c = m_c / N_c`, shape ``(centroid_size, num_classes)``."""
        return self.centroids_sum / self.centroid_counts.unsqueeze(0)

    def kernel(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel against the current centroids.

        Args:
            embeddings: Per-class embeddings of shape
                ``(batch, centroid_size, num_classes)``.

        Returns:
            Per-class kernel values of shape ``(batch, num_classes)``.
        """
        diff = embeddings - self.centroids.unsqueeze(0)
        # Mean over the centroid_size dimension matches the 1 / (n * sigma^2)
        # normalisation used in the reference implementation.
        squared = diff.pow(2).mean(dim=1)
        return torch.exp(-squared / (2.0 * self.length_scale**2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute per-class kernel values for a batch of features."""
        return self.kernel(self.embed(features))

    @torch.no_grad()
    def update_centroids(self, features: torch.Tensor, labels_onehot: torch.Tensor) -> None:
        """EMA-update the centroid statistics from a labelled batch.

        Args:
            features: Feature vectors of shape ``(batch, feature_dim)``.
            labels_onehot: One-hot labels of shape ``(batch, num_classes)``.
        """
        embeddings = self.embed(features)
        class_counts = labels_onehot.sum(dim=0)
        class_sums = torch.einsum("bnc,bc->nc", embeddings, labels_onehot)
        self.centroid_counts.mul_(self.gamma).add_(class_counts, alpha=1.0 - self.gamma)
        self.centroids_sum.mul_(self.gamma).add_(class_sums, alpha=1.0 - self.gamma)


@duq_generator.register(nn.Module)
class TorchDUQPredictor(nn.Module, DUQPredictor[[torch.Tensor], TorchDUQRepresentation]):
    """Torch implementation of an DUQ predictor.

    The traversal replaces the last ``nn.Linear`` (the classification head) with
    ``nn.Identity()``, producing a pure feature encoder. A fresh
    :class:`RBFCentroidHead` is attached as ``centroid_head``. Together the
    predictor produces per-class kernel values used for both classification
    (argmax) and uncertainty (one minus the max).

    Train from scratch with the binary cross-entropy loss on the kernel values
    against one-hot labels, plus a two-sided gradient penalty on the inputs as
    in the reference implementation. After each optimisation step, call
    :meth:`update_centroids` to refresh the EMA centroid statistics.
    """

    encoder: nn.Module
    centroid_head: RBFCentroidHead

    def __init__(
        self,
        model: nn.Module,
        centroid_size: int = 256,
        length_scale: float = 0.1,
        gamma: float = 0.999,
    ) -> None:
        r"""Build the DUQ predictor from a base classification model.

        Args:
            model: Base classification model whose last ``nn.Linear`` defines
                the feature dimension and number of classes.
            centroid_size: Embedding dimension :math:`n`.
            length_scale: RBF kernel length scale :math:`\sigma`.
            gamma: Exponential moving-average decay for the centroid statistics.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(torch_duq_traverser, nn_traverser=nn_traverser),
            init={TRAVERSE_REVERSED: True, HEAD_MODULE: None},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty: ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        self.encoder = encoder
        self.centroid_head = RBFCentroidHead(
            feature_dim=head.in_features,
            num_classes=head.out_features,
            centroid_size=centroid_size,
            length_scale=length_scale,
            gamma=gamma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` and return per-class kernel values of shape ``(batch, num_classes)``."""
        features = self.encoder(x)
        return self.centroid_head(features)

    def update_centroids(self, x: torch.Tensor, labels_onehot: torch.Tensor) -> None:
        """Refresh the EMA centroid statistics from a labeled batch.

        Encoder is briefly switched to evaluation mode so that BatchNorm-style
        layers do not update their running statistics during the centroid pass.

        Args:
            x: Input batch passed through the encoder.
            labels_onehot: One-hot labels of shape ``(batch, num_classes)``.
        """
        encoder_state = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            features = self.encoder(x).detach()
        self.encoder.train(encoder_state)
        self.centroid_head.update_centroids(features, labels_onehot)

    def predict_representation(self, x: torch.Tensor) -> TorchDUQRepresentation:
        """Predict the DUQ representation (per-class kernel values)."""
        return TorchDUQRepresentation(kernel_values=self.forward(x))
