"""Dispatched definitions of the selective prediction metrics."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def risk_coverage_curve(criterion: object, losses: object) -> tuple[object, object, object]:
    """Exact risk-coverage curve of selective prediction.

    Roughly speaking, the curve shows how the loss of the accepted predictions changes as more and more
    uncertain instances are rejected. More precisely, an instance is accepted when its criterion is at most a
    threshold, so instances with a larger criterion (more uncertainty) are rejected first. For every threshold,
    the curve records the coverage, the fraction of accepted instances, and the selective risk, the mean loss
    over the accepted instances, ``sum(losses * accepted) / sum(accepted)``. Being a ratio, the selective risk
    is not monotone in the threshold.

    Tied criterion values form a single step: they are accepted or rejected together, so that the curve does
    not depend on the order of the instances :cite:`jaegerCallReflect2023`. The output nevertheless has a point
    for every instance, which keeps its shape fixed, and every instance of a tied run carries the point at the
    end of that run. For criteria ``[0.1, 0.5, 0.5, 0.9]``, the coverages are ``[0.0, 0.25, 0.75, 0.75, 1.0]``.
    The repeated points add no area.

    The first point is an endpoint at coverage 0 that carries the risk of the first step, following
    :cite:`jaegerCallReflect2023`. Without it, the area would only start at the coverage of the first step, so
    criteria whose first step covers more instances would get a smaller area.

    A NaN in ``criterion`` has no place in the ranking and raises an error. Inside a traced JAX function, such
    as one compiled with ``jax.jit``, the values are unknown, and the check is skipped.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``.

    Returns:
        A tuple containing:
            - coverage: Coverage at every threshold, non-decreasing, of shape ``(n + 1,)``.
            - risk: Selective risk at every threshold, of shape ``(n + 1,)``.
            - thresholds: Criterion values in ascending order, of shape ``(n + 1,)``, starting with ``-inf``
              for the endpoint at coverage 0.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``criterion`` and ``losses`` are not one-dimensional with the same, nonzero length, or if
            ``criterion`` contains NaN.
    """
    msg = f"No risk_coverage_curve implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def aurc(criterion: object, losses: object) -> object:
    """Area under the exact risk-coverage curve.

    The AURC is the trapezoidal area under the selective risk over the coverage of :func:`risk_coverage_curve`;
    lower is better. The selective risk suits a deployed working point, since it is the risk of an accepted
    prediction. Aggregated over all thresholds, however, it overweights failures at low coverage, where few
    instances are accepted. For comparing criteria across thresholds, :func:`augrc` is therefore the better
    choice :cite:`traubOvercomingCommon2024`.

    Inside a step of tied criterion values, the curve is interpolated with the expected selective risk when a
    random part of the tied instances is accepted. The AURC therefore equals the mean AURC over all orders of the
    tied instances. A straight line between the steps, as in :cite:`jaegerCallReflect2023`, lies below this
    expected risk whenever the tied instances are worse than those accepted before them. It cannot be reached by
    any selector, and it would favor criteria that tie many instances, such as coarsened ones. Without ties, the
    two interpolations agree.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``.

    Returns:
        The area under the risk-coverage curve.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``criterion`` and ``losses`` are not one-dimensional with the same, nonzero length, or if
            ``criterion`` contains NaN.
    """
    msg = f"No aurc implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def augrc(criterion: object, losses: object) -> object:
    """Area under the exact generalized risk-coverage curve.

    The generalized risk at a threshold is the selective risk times the coverage, that is, the loss of the
    accepted instances averaged over all instances :cite:`traubOvercomingCommon2024`. In other words, it is the
    risk of a silent failure for any prediction, not only for an accepted one. It improves whenever the ranking
    or the predictions improve, which the AURC does not guarantee. Lower is better.

    For the zero-one loss, the AUGRC has a pairwise reading: it is half the probability that, of two instances
    drawn at random, either both are wrong, or exactly one is wrong and it has the lower criterion (a tie
    counts half). Consequently, it lies in ``[0, 1/2]``.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``.

    Returns:
        The area under the generalized risk-coverage curve.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``criterion`` and ``losses`` are not one-dimensional with the same, nonzero length, or if
            ``criterion`` contains NaN.
    """
    msg = f"No augrc implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def risk_at_coverage(criterion: object, losses: object, coverage: float) -> object:
    """Selective risk at a target coverage.

    Not every coverage can be reached, since tied criterion values are accepted together. The risk is therefore
    taken at the smallest reachable coverage that is at least ``coverage``.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``.
        coverage: Target coverage in ``(0, 1]``.

    Returns:
        The selective risk at that coverage.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``coverage`` is not in ``(0, 1]``, if ``criterion`` and ``losses`` are not
            one-dimensional with the same, nonzero length, or if ``criterion`` contains NaN.
    """
    msg = f"No risk_at_coverage implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def coverage_at_risk(criterion: object, losses: object, risk: float) -> object:
    """Largest coverage whose selective risk is at most a target risk.

    Since the selective risk is not monotone in the threshold, every threshold is checked. A binary search, as in
    :cite:`geifmanSelectiveClassification2017`, can miss the largest coverage that meets the target. Note that
    the result is an empirical evaluation on the given data, not a guarantee for new data.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss of the prediction for every instance, of shape ``(n,)``.
        risk: Target selective risk, at least 0.

    Returns:
        The largest coverage with a selective risk of at most ``risk``, or 0 if there is none.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``risk`` is negative or NaN, if ``criterion`` and ``losses`` are not one-dimensional with
            the same, nonzero length, or if ``criterion`` contains NaN.
    """
    msg = f"No coverage_at_risk implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


def check_inputs(criterion: object, losses: object) -> int:
    """Check that the criterion and the losses are one-dimensional with the same, nonzero length.

    Args:
        criterion: Criterion values.
        losses: Loss values.

    Returns:
        The number of instances.

    Raises:
        ValueError: If the shapes do not match, are not one-dimensional or are empty.
    """
    criterion_shape = tuple(criterion.shape)  # ty:ignore[unresolved-attribute]
    losses_shape = tuple(losses.shape)  # ty:ignore[unresolved-attribute]
    if len(criterion_shape) != 1 or criterion_shape != losses_shape or criterion_shape[0] == 0:
        msg = (
            "criterion and losses must be one-dimensional with the same, nonzero length, "
            f"got shapes {criterion_shape} and {losses_shape}."
        )
        raise ValueError(msg)
    return criterion_shape[0]


def check_no_nan(has_nan: object) -> None:
    """Check that the criterion contains no NaN.

    Args:
        has_nan: Whether the criterion contains a NaN, as a boolean or a zero-dimensional array.

    Raises:
        ValueError: If ``has_nan`` is true.
    """
    if has_nan:
        msg = "criterion must not contain NaN."
        raise ValueError(msg)


def check_coverage(coverage: float) -> None:
    """Check that a target coverage is in ``(0, 1]``.

    Args:
        coverage: Target coverage.

    Raises:
        ValueError: If ``coverage`` is not in ``(0, 1]``.
    """
    if not 0 < coverage <= 1:
        msg = f"coverage must be in (0, 1], got {coverage}."
        raise ValueError(msg)


def check_risk(risk: float) -> None:
    """Check that a target risk is at least 0.

    Args:
        risk: Target selective risk.

    Raises:
        ValueError: If ``risk`` is negative or NaN.
    """
    if not risk >= 0:
        msg = f"risk must be at least 0, got {risk}."
        raise ValueError(msg)
