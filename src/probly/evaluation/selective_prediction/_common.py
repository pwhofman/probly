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


# The full-coverage risk is the mean loss. It tells a better classifier from a better ranking.
_DEFAULT_METRICS = ("aurc", "augrc", "risk@1")
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
    with ``jax.jit``. Inside such a function, call the metric functions directly.

    Args:
        criterion: Criterion values of shape ``(n,)``, as an array of any supported backend or a list. Larger
            values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``, as an array or a list.
        metrics: The metrics to compute.

            - None or ``"all"``: the default set ``"aurc"``, ``"augrc"`` and ``"risk@1"``. Other working points
              are not included and must be named. The risk at full coverage is the mean loss. A ranking metric
              alone cannot tell a better classifier from a better ranking of its errors, so it is reported next
              to them.
            - A name or a list of names from ``"aurc"``, ``"augrc"``, ``"risk@<coverage>"`` (the selective
              risk at a coverage, see :func:`~probly.metrics.selective_prediction.risk_at_coverage`) and
              ``"coverage@<risk>"`` (the largest coverage with at most that risk, see
              :func:`~probly.metrics.selective_prediction.coverage_at_risk`). The target is a fraction or a
              percentage, for example ``"risk@0.8"`` or ``"risk@80%"``.

    Returns:
        A dictionary of floats with the following keys.

        - ``"<name>"``: the value of each requested metric.
        - ``"<name>:coverage"`` for ``"risk@<coverage>"``: the coverage that was actually used. It is above the
          target when the target is not a reachable coverage, and it can be far above it when many criterion
          values are tied.
        - ``"<name>:risk"`` for ``"coverage@<risk>"``: the selective risk at that coverage, or NaN if no
          coverage meets the target.

        The default set thus returns ``"aurc"``, ``"augrc"``, ``"risk@1"`` and ``"risk@1:coverage"``.

    Raises:
        ValueError: If a metric name is unknown or its target value is invalid, and for the invalid inputs
            listed in the metric functions.
        NotImplementedError: If no metric implementation is registered for the type of ``criterion``.
    """
    if isinstance(criterion, (list, tuple)):
        criterion = np.asarray(criterion)
    if isinstance(losses, (list, tuple)):
        losses = np.asarray(losses)

    if metrics is None or (isinstance(metrics, str) and metrics.lower().strip() == "all"):
        names = list(_DEFAULT_METRICS)
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
