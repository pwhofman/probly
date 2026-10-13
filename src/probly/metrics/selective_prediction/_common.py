"""Dispatched definitions of the selective prediction metrics."""

from __future__ import annotations

from flextype import flexdispatch


@flexdispatch
def risk_coverage_curve(criterion: object, losses: object) -> tuple[object, object, object]:
    """Exact (unbinned) risk-coverage curve of selective prediction.

    The curve shows how the risk of a predictor changes as it abstains on more of its least confident instances.
    Concretely, an instance is accepted when its criterion is at most a threshold, so larger criterion values, which
    indicate more uncertainty, are rejected first. For every threshold, the curve gives the coverage, that is, the
    fraction of accepted instances, and the selective risk, the mean loss of the accepted instances. Note that the
    selective risk need not be monotone in the coverage.

    Tied criterion values cannot be separated by a threshold and are accepted together, so a run of ties forms one
    step of the curve, and the curve does not depend on the order of the instances :cite:`jaegerCallReflect2023`.
    To keep the static shape ``(n + 1,)`` that ``jax.jit`` requires, every instance of a run repeats the point at
    the end of the run: criteria ``[0.1, 0.5, 0.5, 0.9]`` give the coverages ``[0, 0.25, 0.75, 0.75, 1]``. The
    first point is an endpoint at coverage 0, where the selective risk is undefined; by convention, it carries the
    risk of the first step :cite:`jaegerCallReflect2023`. Without it, an area would start at the coverage of the
    first step and thus favor criteria with a large first step.

    A criterion of NaN or ``-inf``, which is reserved for the endpoint, and a non-finite loss raise an error, while
    a criterion of ``+inf`` is allowed. Under ``jax.jit``, these checks are skipped, since the values are unknown
    during tracing.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss per instance, of shape ``(n,)``, on the backend of ``criterion`` or as a NumPy array.

    Returns:
        Tuple of arrays on the backend of ``criterion``:
            - coverage: Coverage at every threshold, non-decreasing, of shape ``(n + 1,)``.
            - risk: Selective risk at every threshold, of shape ``(n + 1,)``.
            - thresholds: Criterion values in ascending order, of shape ``(n + 1,)``, starting with ``-inf``
              for the endpoint.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``criterion`` and ``losses`` are not one-dimensional with the same, nonzero length, if
            ``criterion`` contains NaN or ``-inf``, or if ``losses`` contains NaN or an infinite value.
    """
    msg = f"No risk_coverage_curve implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def aurc(criterion: object, losses: object) -> object:
    """Area under the risk-coverage curve, averaged over the orders of tied criterion values.

    Lower is better. The selective risk is the natural quantity at a deployed working point. Aggregated over all
    thresholds, however, it overweights failures at low coverage, where a single failure is averaged over few
    instances, so :func:`augrc` is the better choice for comparing criteria over all working points
    :cite:`traubOvercomingCommon2024`.

    With ties, the AURC is not the trapezoid over :func:`risk_coverage_curve`. Inside a step, it follows the
    expected selective risk when a random part of the run is accepted; in other words, it is the mean of the AURC
    over all orders of the tied instances. The straight line inside a step :cite:`jaegerCallReflect2023`, in
    contrast, cannot be reached by a selector that uses only the criterion. It also lies below the expected risk
    when the step is worse than what was accepted before it, so it favors criteria with many ties, such as
    coarsened ones. Without ties, both areas agree.

    For losses in ``[0, 1]``, the AURC lies in ``[0, 1]``. Without ties, it equals the per-instance mean
    ``(1/n) sum_k SR_k`` :cite:`geifmanBiasReduced2019`, where ``SR_k`` is the selective risk of the ``k`` most
    confident instances, plus ``(SR_1 - SR_n) / (2n)``, which vanishes as ``n`` grows for bounded losses.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss per instance, of shape ``(n,)``, on the backend of ``criterion`` or as a NumPy array.

    Returns:
        The AURC, a float for NumPy inputs and a zero-dimensional array otherwise.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: For the invalid inputs listed in :func:`risk_coverage_curve`.
    """
    msg = f"No aurc implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def augrc(criterion: object, losses: object) -> object:
    """Area under the exact generalized risk-coverage curve.

    Roughly speaking, the generalized risk is the risk of a silent failure, that is, of an instance that is
    accepted and then fails :cite:`traubOvercomingCommon2024`. More precisely, it is the selective risk times the
    coverage, or the loss of the accepted instances averaged over all instances. Lower is better. Unlike the
    selective risk, its expectation is linear inside a step, so the trapezoid over :func:`risk_coverage_curve`
    already equals the mean over all orders of the tied instances.

    For the zero-one loss, the AUGRC decreases as both the accuracy and the AUROC of the criterion as a failure
    detector increase, whereas the AURC need not :cite:`traubOvercomingCommon2024`. Indeed, the AUGRC is then half
    the probability that, of two instances drawn independently, both are wrong, or exactly one is wrong and it has
    the lower criterion, where ties count half. For a general loss, it is half the mean loss plus a concordance
    term that rewards ranking larger losses toward rejection and to which tied pairs contribute 0. For losses in
    ``[0, 1]``, it lies in ``[0, 1/2]``.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss per instance, of shape ``(n,)``, on the backend of ``criterion`` or as a NumPy array.

    Returns:
        The AUGRC, a float for NumPy inputs and a zero-dimensional array otherwise.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: For the invalid inputs listed in :func:`risk_coverage_curve`.
    """
    msg = f"No augrc implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def risk_at_coverage(criterion: object, losses: object, coverage: float) -> tuple[object, object]:
    """Selective risk at the smallest reachable coverage that is at least a target.

    Not every coverage is reachable: a threshold accepts ``k`` of the ``n`` instances, and ties skip some values
    of ``k``. The target is therefore rounded up to a coverage of :func:`risk_coverage_curve`, and the result is
    the risk there, not the minimum risk over larger coverages. Under ``jax.jit``, ``coverage`` must be a static
    argument.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss per instance, of shape ``(n,)``, on the backend of ``criterion`` or as a NumPy array.
        coverage: Target coverage in ``(0, 1]``.

    Returns:
        Tuple of floats for NumPy inputs, of zero-dimensional arrays otherwise:
            - risk: The selective risk at the realized coverage.
            - coverage: The realized coverage. With many ties, it can be far above the target, so two criteria
              are comparable only at equal realized coverages.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``coverage`` is not in ``(0, 1]``, or for the invalid inputs listed in
            :func:`risk_coverage_curve`.
    """
    msg = f"No risk_at_coverage implementation registered for type {type(criterion)}"
    raise NotImplementedError(msg)


@flexdispatch
def coverage_at_risk(criterion: object, losses: object, risk: float) -> tuple[object, object]:
    """Largest coverage whose selective risk is at most a target.

    Since the selective risk need not be monotone in the coverage, every threshold is checked; a binary search,
    as in :cite:`geifmanSelectiveClassification2017`, can miss the largest such coverage. Note that the result is
    an empirical working point and carries no guarantee for new data. Under ``jax.jit``, ``risk`` must be a static
    argument.

    Args:
        criterion: Criterion values of shape ``(n,)``. Larger values are rejected first.
        losses: Loss per instance, of shape ``(n,)``, on the backend of ``criterion`` or as a NumPy array.
        risk: Target selective risk, at least 0.

    Returns:
        Tuple of floats for NumPy inputs, of zero-dimensional arrays otherwise:
            - coverage: The largest coverage that meets the target, or 0 if there is none.
            - risk: The realized selective risk at that coverage, or NaN if there is none.

    Raises:
        NotImplementedError: If no implementation is registered for the type of ``criterion``.
        ValueError: If ``risk`` is negative or NaN, or for the invalid inputs listed in
            :func:`risk_coverage_curve`.
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


def check_finite_losses(all_finite: object) -> None:
    """Check that the losses contain neither NaN nor an infinite value.

    A NaN loss would silently drop the affected points from the working points, and an infinite loss would make
    the AURC NaN.

    Args:
        all_finite: Whether all losses are finite, as a boolean or a zero-dimensional array.

    Raises:
        ValueError: If ``all_finite`` is false.
    """
    if not all_finite:
        msg = "losses must be finite, without NaN or inf."
        raise ValueError(msg)


def check_no_negative_infinity(has_negative_infinity: object) -> None:
    """Check that the criterion contains no ``-inf``.

    The threshold ``-inf`` marks the endpoint at coverage 0, so an instance with that criterion value would be
    accepted at the endpoint.

    Args:
        has_negative_infinity: Whether the criterion contains ``-inf``, as a boolean or a zero-dimensional array.

    Raises:
        ValueError: If ``has_negative_infinity`` is true.
    """
    if has_negative_infinity:
        msg = "criterion must not contain -inf, which is the threshold of the endpoint at coverage 0."
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
