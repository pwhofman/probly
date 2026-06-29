"""Scoring rules identified by their per-label loss vector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from flextype import flexdispatch


class ScoringRule(ABC):
    """Scoring rule, defined by its per-label loss vector.

    The single primitive is :meth:`loss`, mapping a predicted distribution
    ``theta_hat`` of shape ``(..., K)`` to the loss vector
    ``[l(theta_hat, 1), ..., l(theta_hat, K)]``, from which
    :class:`SecondOrderScoringRuleDecomposition` derives the total, aleatoric, and
    epistemic measures. The decomposition runs for any rule but is a valid
    uncertainty decomposition (epistemic uncertainty ``>= 0``) only for proper
    rules; the built-ins are proper. Built-ins support NumPy and PyTorch; a custom
    rule only needs the backend(s) it is used with.

    Example:
        A custom pseudo-spherical score of order 3, with ``G(theta) = 1 - ||theta||_3``::

            class PseudoSphericalLoss(ScoringRule):
                def loss(self, probabilities):
                    norm = torch.sum(probabilities**3, dim=-1, keepdim=True) ** (1 / 3)
                    return 1.0 - probabilities**2 / norm**2

            decomposition = SecondOrderScoringRuleDecomposition(sample, PseudoSphericalLoss())
    """

    @abstractmethod
    def loss[ArrayT](self, probabilities: ArrayT) -> ArrayT:
        """Return the per-label loss vector ``[l(theta_hat, 1), ..., l(theta_hat, K)]``.

        Args:
            probabilities: Predicted categorical probabilities of shape ``(..., K)``.

        Returns:
            The loss vector of shape ``(..., K)`` in the same backend as the input.
        """


@flexdispatch
def _log_loss_vector[ArrayT](probabilities: ArrayT) -> ArrayT:
    """Per-label log loss vector ``-log(theta_hat)``."""
    msg = f"LogLoss is not supported for arrays of type {type(probabilities)}."
    raise NotImplementedError(msg)


@flexdispatch
def _brier_loss_vector[ArrayT](probabilities: ArrayT) -> ArrayT:
    """Per-label Brier loss vector ``||theta_hat||^2 - 2 theta_hat + 1``."""
    msg = f"BrierLoss is not supported for arrays of type {type(probabilities)}."
    raise NotImplementedError(msg)


@flexdispatch
def _zero_one_loss_vector[ArrayT](probabilities: ArrayT) -> ArrayT:
    """Per-label zero-one loss vector ``1 - onehot(argmax theta_hat)``."""
    msg = f"ZeroOneLoss is not supported for arrays of type {type(probabilities)}."
    raise NotImplementedError(msg)


@flexdispatch
def _spherical_loss_vector[ArrayT](probabilities: ArrayT) -> ArrayT:
    """Per-label spherical loss vector ``1 - theta_hat / ||theta_hat||_2``."""
    msg = f"SphericalLoss is not supported for arrays of type {type(probabilities)}."
    raise NotImplementedError(msg)


@dataclass(frozen=True, slots=True)
class LogLoss(ScoringRule):
    """Log loss ``l(theta_hat, y) = -log(theta_hat_y)``; yields the Shannon-entropy decomposition.

    The logarithmic score is unbounded, so :meth:`loss` is ``+inf`` where a
    probability is zero. The decomposition applies the ``0 log 0 = 0`` convention,
    so its generalized entropy equals the (finite) Shannon entropy.
    """

    def loss[ArrayT](self, probabilities: ArrayT) -> ArrayT:
        return _log_loss_vector(probabilities)


@dataclass(frozen=True, slots=True)
class BrierLoss(ScoringRule):
    """Brier loss ``l(theta_hat, y) = sum_k (theta_hat_k - [k == y])^2``; yields the Gini decomposition."""

    def loss[ArrayT](self, probabilities: ArrayT) -> ArrayT:
        return _brier_loss_vector(probabilities)


@dataclass(frozen=True, slots=True)
class ZeroOneLoss(ScoringRule):
    """Zero-one loss ``l(theta_hat, y) = 1 - [argmax_k theta_hat_k == y]``."""

    def loss[ArrayT](self, probabilities: ArrayT) -> ArrayT:
        return _zero_one_loss_vector(probabilities)


@dataclass(frozen=True, slots=True)
class SphericalLoss(ScoringRule):
    """Spherical loss ``l(theta_hat, y) = 1 - theta_hat_y / ||theta_hat||_2``."""

    def loss[ArrayT](self, probabilities: ArrayT) -> ArrayT:
        return _spherical_loss_vector(probabilities)
