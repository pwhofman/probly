"""Torch quantification of DDU representations."""

from __future__ import annotations

import torch

from probly.representation.ddu.torch import TorchDDURepresentation

from ._common import ddu_epistemic_uncertainty


@ddu_epistemic_uncertainty.register(TorchDDURepresentation)
def torch_ddu_epistemic_uncertainty(representation: TorchDDURepresentation) -> torch.Tensor:
    r"""Negative log GMM marginal density :math:`-\log q(z)` for torch tensors.

    The DDU epistemic uncertainty score is the negative log of the marginal
    density under the class-conditional GMM fitted to training features:

    .. math::

        -\log q(z) = -\log \sum_c \pi_c \mathcal{N}(z;\,\mu_c,\Sigma_c)

    where ``representation.densities`` contains
    :math:`\log \pi_c + \log \mathcal{N}(z;\,\mu_c,\Sigma_c)` for each class
    ``c``, so the marginal log-density reduces to a ``logsumexp`` over classes.

    Reference: Mukhoti et al., "Deep Deterministic Uncertainty", CVPR 2023
    (https://arxiv.org/abs/2102.11582), Section 3.

    Args:
        representation: DDU representation produced by a DDU predictor.

    Returns:
        Per-sample epistemic uncertainty scores of shape ``(...,)``.
    """
    return -torch.logsumexp(representation.densities, dim=-1)
