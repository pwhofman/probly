"""Common helpers for representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation import Representation


@flexdispatch
def compute_mean_probs[T](rep: Representation[T]) -> T:
    """Compute the mean softmax probabilities across samples from sampler outputs."""
    msg = f"No mean probs computation registered for type {type(rep)}"
    raise NotImplementedError(msg)
