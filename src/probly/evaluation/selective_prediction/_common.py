"""Dispatched definition of the selective prediction evaluation task."""

from __future__ import annotations

from flextype import flexdispatch
import numpy as np

from probly.metrics.selective_prediction import augrc, aurc, coverage_at_risk, risk_at_coverage


@flexdispatch
def selective_prediction(criterion: object, losses: object, n_bins: int = 50) -> tuple[object, object]:
    """Selective prediction downstream task for evaluation.

    Perform selective prediction based on criterion and losses. The criterion is used to sort the losses.
    In line with uncertainty literature the sorting is done in descending order, i.e. the losses with the
    largest criterion are rejected first. Ties are broken by input position: among equal criterion values,
    earlier instances are rejected first, so results are deterministic across backends.

    The curve is binned, and tied criterion values are split by position. The exact risk-coverage curve, in
    which they form a single step, is :func:`probly.metrics.selective_prediction.risk_coverage_curve`, and its
    area is :func:`probly.metrics.selective_prediction.aurc`.

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


_METRICS = ("aurc", "augrc")
_WORKING_POINTS = {"risk": risk_at_coverage, "coverage": coverage_at_risk}
# The second output of a working point, reported next to its first.
_COMPANIONS = {"risk": "coverage", "coverage": "risk"}


def evaluate_selective_prediction(
    criterion: object,
    losses: object,
    metrics: str | list[str] | None = None,
) -> dict[str, float]:
    """Evaluate selective prediction with several metrics at once.

    The metrics come from :mod:`probly.metrics.selective_prediction`, and their values are converted to
    Python floats. For this reason, the function cannot be used inside a traced function, such as one compiled
    with ``jax.jit``; there, the metric functions are called directly.

    Args:
        criterion: Criterion values of shape ``(n,)``, as an array of any supported backend or a list. Larger
            values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``, as an array or a list.
        metrics: The metrics to compute.

            - None or ``"all"``: ``"aurc"`` and ``"augrc"``.
            - A name or a list of names from ``"aurc"``, ``"augrc"``, ``"risk@<coverage>"`` (the selective
              risk at a coverage, see :func:`~probly.metrics.selective_prediction.risk_at_coverage`) and
              ``"coverage@<risk>"`` (the largest coverage with at most that risk, see
              :func:`~probly.metrics.selective_prediction.coverage_at_risk`). The target is a fraction or a
              percentage, for example ``"risk@0.8"`` or ``"risk@80%"``.

    Returns:
        A dictionary mapping each requested metric name to its value. A working point also reports the other
        output of its function under the key ``"<name>:coverage"`` for ``"risk@<coverage>"``, the coverage that
        was actually used, which can be above the target when criterion values are tied, and ``"<name>:risk"``
        for ``"coverage@<risk>"``, the selective risk at that coverage (NaN if no coverage meets the target).

    Raises:
        ValueError: If a metric name is unknown or its target value is invalid.
    """
    if isinstance(criterion, (list, tuple)):
        criterion = np.asarray(criterion)
    if isinstance(losses, (list, tuple)):
        losses = np.asarray(losses)

    if metrics is None or (isinstance(metrics, str) and metrics.lower().strip() == "all"):
        names = list(_METRICS)
    elif isinstance(metrics, str):
        names = [metrics]
    else:
        names = list(metrics)

    results: dict[str, float] = {}
    for name in names:
        key = name.lower().strip()
        if key == "aurc":
            value = aurc(criterion, losses)
        elif key == "augrc":
            value = augrc(criterion, losses)
        else:
            base, _, target = key.partition("@")
            target = target.strip()
            try:
                function = _WORKING_POINTS[base.strip()]
                target_value = float(target[:-1]) / 100 if target.endswith("%") else float(target)
            except (KeyError, ValueError):
                msg = (
                    f"Unknown metric {name!r}. Available: 'aurc', 'augrc', 'risk@<coverage>' and "
                    "'coverage@<risk>', for example 'risk@0.8' or 'risk@80%'."
                )
                raise ValueError(msg) from None
            value, companion = function(criterion, losses, target_value)
            results[name] = float(value)  # ty:ignore[invalid-argument-type]
            results[f"{name}:{_COMPANIONS[base.strip()]}"] = float(companion)  # ty:ignore[invalid-argument-type]
            continue
        results[name] = float(value)  # ty:ignore[invalid-argument-type]
    return results
