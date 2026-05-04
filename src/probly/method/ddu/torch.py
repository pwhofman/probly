"""Torch implementation of the DDU transformation."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING, ClassVar
import warnings

import torch
from torch import nn
from torch.nn.utils import parametrize

from probly.layers.torch import GaussianMixtureHead, SNCoeffParametrization
from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import (
    DDUPredictor,
    DDURepresentation,
    create_ddu_representation,
    ddu_generator,
    negative_log_density,
)

if TYPE_CHECKING:
    from torch import Tensor


@create_ddu_representation.register(TorchCategoricalDistribution)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchDDURepresentation(DDURepresentation, TorchAxisProtected):
    """DDU representation backed by torch tensors."""

    softmax: TorchCategoricalDistribution
    densities: Tensor
    protected_axes: ClassVar[dict[str, int]] = {"softmax": 0, "densities": 1}


@negative_log_density.register(torch.Tensor)
def torch_negative_log_density(densities: torch.Tensor) -> torch.Tensor:
    """Convert class-weighted log densities to negative GMM log density."""
    return -torch.logsumexp(densities, dim=-1)


SN_COEFF = GlobalVariable[float]("SN_COEFF", default=3.0)
HAS_RESIDUAL = GlobalVariable[bool]("HAS_RESIDUAL", default=False)
HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("HEAD_MODULE", default=None)


@singledispatch_traverser
def residual_detection_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Detect residual connections by FX-tracing non-leaf modules for add operations.

    Sets HAS_RESIDUAL to True in the traversal state as soon as an addition
    node is found in any submodule's computation graph.
    """
    if state[HAS_RESIDUAL] or not list(obj.named_children()):
        return obj, state
    traced = torch.fx.symbolic_trace(obj)
    for node in traced.graph.nodes:
        if node.op == "call_function" and node.target in (torch.add, operator.add, operator.iadd):
            state[HAS_RESIDUAL] = True
            break
        if node.op == "call_method" and node.target == "add":
            state[HAS_RESIDUAL] = True
            break
    return obj, state


@singledispatch_traverser
def torch_ddu_traverser(obj: nn.Module, state: State) -> tuple[nn.Module, State]:
    """Default handler: return module unchanged."""
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.Linear, state: State) -> tuple[nn.Module, State]:
    """Replace the classification head with Identity; apply spectral normalization to others.

    With TRAVERSE_REVERSED, the last Linear layer (the head) is encountered
    first. It is stored in HEAD_MODULE and replaced with ``nn.Identity()`` so
    the resulting model becomes a pure feature encoder.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return nn.Identity(), state
    parametrize.register_parametrization(obj, "weight", SNCoeffParametrization(state[SN_COEFF], obj.weight))
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.Conv2d, state: State) -> tuple[nn.Module, State]:
    """Apply spectral normalization to Conv2d layers.

    Stride-1x1 convolutions (typical residual-branch downsampling) are first
    replaced with an AvgPool2d followed by a stride-1 Conv2d to remove aliasing,
    then spectral normalization is applied to the new Conv2d.
    """
    stride = obj.stride[0] if isinstance(obj.stride, (tuple, list)) else obj.stride
    kernel = obj.kernel_size[0] if isinstance(obj.kernel_size, (tuple, list)) else obj.kernel_size
    if stride > 1 and kernel == 1:
        avg_pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        new_conv = nn.Conv2d(
            obj.in_channels,
            obj.out_channels,
            kernel_size=1,
            stride=1,
            padding=obj.padding if isinstance(obj.padding, int) else obj.padding[0],
            bias=obj.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(obj.weight)
            if obj.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(obj.bias)
        parametrize.register_parametrization(
            new_conv, "weight", SNCoeffParametrization(state[SN_COEFF], new_conv.weight)
        )
        return nn.Sequential(avg_pool, new_conv), state
    parametrize.register_parametrization(obj, "weight", SNCoeffParametrization(state[SN_COEFF], obj.weight))
    return obj, state


@torch_ddu_traverser.register
def _(obj: nn.ReLU, state: State) -> tuple[nn.Module, State]:  # noqa: ARG001
    """Replace ReLU with LeakyReLU(0.01)."""
    return nn.LeakyReLU(negative_slope=0.01, inplace=False), state


@torch_ddu_traverser.register
def _(obj: nn.ReLU6, state: State) -> tuple[nn.Module, State]:  # noqa: ARG001
    """Replace ReLU6 with LeakyReLU(0.01)."""
    return nn.LeakyReLU(negative_slope=0.01, inplace=False), state


@ddu_generator.register(nn.Module)
class TorchDDUPredictor(nn.Module, DDUPredictor[[torch.Tensor], TorchDDURepresentation]):
    """Torch version of a DDU predictor.

    The traversal replaces the last ``nn.Linear`` (the classification head) with
    ``nn.Identity()``, producing a pure feature encoder.  The head is stored
    separately so the predictor has three clean, independent components:

    - ``encoder``: the spectrally-normalised backbone (head replaced with Identity).
    - ``classification_head``: the original last Linear layer.
    - ``density_head``: a :class:`GaussianMixtureHead` fitted post-training.

    Call ``density_head.fit(features, labels)`` after training to initialise
    the GMM parameters.
    """

    encoder: nn.Module
    classification_head: nn.Linear
    density_head: GaussianMixtureHead

    def __init__(self, model: nn.Module, sn_coeff: float = 3.0) -> None:
        """Build the three-component DDU predictor from a base model.

        Args:
            model: Base classification model to be transformed.
            sn_coeff: Lipschitz coefficient for spectral normalization.
        """
        super().__init__()
        encoder, state = traverse_with_state(
            model,
            nn_compose(residual_detection_traverser, torch_ddu_traverser, nn_traverser=nn_traverser),
            init={HAS_RESIDUAL: False, TRAVERSE_REVERSED: True, SN_COEFF: sn_coeff, HEAD_MODULE: None},
        )
        head: nn.Linear | None = state[HEAD_MODULE]  # ty: ignore[invalid-assignment]
        if head is None:
            msg = "No nn.Linear layer found in the model; cannot identify a classification head."
            raise ValueError(msg)

        if not state[HAS_RESIDUAL]:
            warnings.warn(
                "No residual connections detected in the given model. DDU is designed for models "
                "with residual connections (e.g. ResNets). Without them, features may collapse "
                "under spectral normalization and density estimates will be unreliable. "
                "Please consider to use a different method or to add residual connections to your model.",
                UserWarning,
                stacklevel=2,
            )

        self.encoder = encoder
        self.classification_head = head
        self.density_head = GaussianMixtureHead(head.out_features, head.in_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode features, classify, and score density.

        Args:
            x: Input tensor passed to the encoder.

        Returns:
            A 2-tuple of ``(logits, density_scores)`` where both have shape
            ``(N, num_classes)``.
        """
        features = self.encoder(x)
        logits = self.classification_head(features)
        densities = self.density_head(features)
        return logits, densities

    def fit_density_head(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit the GMM density head to the given features and labels."""
        encoder_state = self.encoder.training  # save state
        density_state = self.density_head.training
        self.encoder.eval()
        self.density_head.train()
        features = self.encoder(x).detach()
        self.density_head.fit(features, labels)
        self.encoder.train(encoder_state)  # restore state
        self.density_head.train(density_state)

    def predict_representation(self, x: torch.Tensor) -> TorchDDURepresentation:
        """Predict the DDU representation (logits and density scores)."""
        logits, densities = self.forward(x)

        return TorchDDURepresentation(
            TorchCategoricalDistribution(torch.softmax(logits, dim=-1)),
            densities,
        )
