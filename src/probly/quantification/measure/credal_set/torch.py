"""Uncertainty measures for torch-based credal sets."""

from __future__ import annotations

import torch

from probly.representation.credal_set.torch import TorchConvexCredalSet
from probly.utils.torch import torch_entropy

from ._common import LogBase, _apply_base, _constant_like, _machine_epsilon, _take_along_axis, upper_entropy

_LBFGS_ITERS = 128


@_take_along_axis.register(torch.Tensor)
def _torch_take_along_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather the entries at the indices along the last dimension."""
    return values.gather(-1, indices)


@_constant_like.register(torch.Tensor)
def _torch_constant_like(reference: torch.Tensor, values: object) -> torch.Tensor:
    """Create a torch tensor with the dtype and on the device of the reference."""
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


@_machine_epsilon.register(torch.Tensor)
def _torch_machine_epsilon(values: torch.Tensor) -> float:
    """Return the machine epsilon of the dtype of a floating torch tensor."""
    return torch.finfo(values.dtype).eps


@upper_entropy.register(TorchConvexCredalSet)
def torch_convex_upper_entropy(
    credal_set: TorchConvexCredalSet,
    base: LogBase = None,
    *,
    return_distribution: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute the upper entropy of a convex hull credal set.

    Maximize entropy over conv(vertices) via L-BFGS on softmax weights.

    Since entropy is concave the maximum over a convex hull may be in the
    interior; L-BFGS on the unconstrained softmax parameterization handles this.
    """
    vertices = credal_set.tensor.probabilities
    batch_shape = vertices.shape[:-2]
    *_, n_vertices, n_classes = vertices.shape
    flat_v = vertices.detach().reshape(-1, n_vertices, n_classes)
    n = flat_v.shape[0]

    w_logits = flat_v.new_zeros(n, n_vertices).requires_grad_(True)
    opt = torch.optim.LBFGS([w_logits], max_iter=_LBFGS_ITERS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        p = (w_logits.softmax(-1).unsqueeze(-1) * flat_v).sum(-2)
        loss = -torch_entropy(p).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        p = (w_logits.softmax(-1).unsqueeze(-1) * flat_v).sum(-2)
    result = _apply_base(torch_entropy(p).reshape(batch_shape), credal_set.num_classes, base)
    if return_distribution:
        return result, p.reshape(*batch_shape, n_classes)
    return result
