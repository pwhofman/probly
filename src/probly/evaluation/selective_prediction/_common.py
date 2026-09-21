"""Dispatched definition of the selective prediction evaluation task."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def selective_prediction(criterion: object, losses: object, n_bins: int = 50) -> tuple[object, object]:
    """Selective prediction downstream task for evaluation.

    Perform selective prediction based on criterion and losses. The criterion is used to sort the losses.
    In line with uncertainty literature the sorting is done in descending order, i.e. the losses with the
    largest criterion are rejected first.

    Args:
        criterion: Criterion values of shape (n_instances,).
        losses: Loss values of shape (n_instances,).
        n_bins: Number of bins.

    Returns:
        A tuple containing:
            - aurc: Area under the risk / loss curve.
            - bin_losses: Loss per bin of shape (n_bins,).

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
    """
    msg = f"No selective_prediction implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)
