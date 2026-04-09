"""Torch implementation of Deep Deterministic Uncertainty (DDU)."""

from __future__ import annotations

import operator
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from probly.traverse_nn import nn_compose, nn_traverser
from pytraverse import TRAVERSE_REVERSED, GlobalVariable, State, singledispatch_traverser, traverse_with_state

from ._common import ddu_generator

SN_COEFF = GlobalVariable[float]("SN_COEFF", default=3.0)
HAS_RESIDUAL = GlobalVariable[bool]("HAS_RESIDUAL", default=False)
HEAD_MODULE: GlobalVariable[nn.Module | None] = GlobalVariable("HEAD_MODULE", default=None)


class _SNCoeffParametrization(nn.Module):
    """Weight parametrization that clips the spectral norm to at most coeff.

    Uses one step of power iteration to estimate the spectral norm and then
    scales the weight so that ||W||_2 <= coeff.
    """

    def __init__(self, coeff: float, weight: torch.Tensor) -> None:
        """Initialize the parametrization.

        Args:
            coeff: Maximum allowed spectral norm.
            weight: Reference weight tensor used for shape and device.
        """
        super().__init__()
        self.coeff = coeff
        self._u: torch.Tensor
        self.register_buffer("_u", F.normalize(torch.randn(weight.shape[0], device=weight.device), dim=0))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Rescale weight so that its spectral norm is at most coeff.

        Args:
            weight: The original weight tensor.

        Returns:
            Rescaled weight tensor.
        """
        w_mat = weight.reshape(weight.shape[0], -1)
        with torch.no_grad():
            v = F.normalize(w_mat.T @ self._u, dim=0)
            u = F.normalize(w_mat @ v, dim=0)
            self._u.copy_(u)
        sigma = u @ (w_mat @ v)
        return weight * (self.coeff / sigma.clamp(min=self.coeff))

    def right_inverse(self, weight: torch.Tensor) -> torch.Tensor:
        """Identity right-inverse to support state_dict loading.

        Args:
            weight: The parametrized weight.

        Returns:
            The weight unchanged.
        """
        return weight


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
    """Skip the classification head; apply spectral normalization to all other Linear layers.

    With TRAVERSE_REVERSED, the last Linear layer (the head) is encountered
    first, so HEAD_SKIPPED acts as a once-off guard.
    """
    if state[HEAD_MODULE] is None:
        state[HEAD_MODULE] = obj
        return obj, state
    parametrize.register_parametrization(obj, "weight", _SNCoeffParametrization(state[SN_COEFF], obj.weight))
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
            new_conv, "weight", _SNCoeffParametrization(state[SN_COEFF], new_conv.weight)
        )
        return nn.Sequential(avg_pool, new_conv), state
    parametrize.register_parametrization(obj, "weight", _SNCoeffParametrization(state[SN_COEFF], obj.weight))
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
def generate_torch_ddu(model: nn.Module, sn_coeff: float = 3.0) -> nn.Module:
    """Build a torch DDU model based on :cite:`mukhotiDeepDeterministicUncertainty2023`.

    Args:
        model: The torch model to be transformed.
        sn_coeff: Lipschitz coefficient for spectral normalization. Default is 3.
    """
    model, state = traverse_with_state(
        model,
        nn_compose(residual_detection_traverser, torch_ddu_traverser, nn_traverser=nn_traverser),
        init={HAS_RESIDUAL: False, TRAVERSE_REVERSED: True, SN_COEFF: sn_coeff, HEAD_MODULE: None},
    )
    if (head := state[HEAD_MODULE]) is not None:
        # Store as a plain attribute (not a registered submodule) so that the
        # feature_head can be looked up by name without duplicating parameters.
        object.__setattr__(model, "feature_head", head)
    if not state[HAS_RESIDUAL]:
        warnings.warn(
            "No residual connections detected in the given model. DDU is designed for models "
            "with residual connections (e.g. ResNets). Without them, features may collapse "
            "under spectral normalization and density estimates will be unreliable. Please consider to use a different"
            " method or to add residual connections to your model.",
            UserWarning,
            stacklevel=2,
        )
    return model
