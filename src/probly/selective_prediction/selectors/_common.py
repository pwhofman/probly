"""Selectors: rules that accept or reject predictions based on their uncertainty criterion."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import math
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast, override

from flextype import flexdispatch, switch
import numpy as np
from scipy.special import betaincinv

from probly.quantification import Notion, TotalUncertainty, notion_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from ndlike.array_like import ArrayLikeWithoutSize

    from probly.quantification.notion import NotionKey, NotionName

type SelectorName = Literal["threshold", "coverage", "risk"]


class Selector(ABC):
    """Base class for rules that accept or reject predictions based only on their uncertainty criterion.

    A selector does not see the model, so it also works on scores computed for any other model. The criterion is a
    one-dimensional array with one value per instance. As in
    :obj:`~probly.metrics.selective_prediction.risk_coverage_curve`, higher values mean more uncertain and are
    rejected first, so a confidence score has to be negated. A NaN criterion is rejected.

    A subclass overrides :meth:`select`. A rule fitted on data implements :class:`~probly.calibrator.Calibrator` on
    criterion arrays, sets :attr:`takes_losses` if ``calibrate`` also takes the losses of the calibration predictions,
    and has an ``is_calibrated`` attribute, which :class:`~probly.selective_prediction.SelectivePredictor` checks
    before the forward pass; while it is ``False``, :meth:`select` raises a ``ValueError``.

    Attributes:
        takes_losses: Whether ``calibrate`` takes the losses of the calibration predictions as its second argument.
        notion: The notion class or the function that gives the criterion, read by
            :class:`~probly.selective_prediction.SelectivePredictor`. Defaults to the total uncertainty.
        threshold: The fixed or fitted threshold on the criterion, or ``None`` if there is none (yet).
        bound: The certified bound on the risk, or ``None`` if there is none (yet).
    """

    takes_losses: ClassVar[bool] = False
    notion: type[Notion] | Callable[[Any], Any] = TotalUncertainty
    threshold: float | None = None
    bound: float | None = None

    @abstractmethod
    def select(self, uncertainty: ArrayLikeWithoutSize) -> ArrayLikeWithoutSize:
        """Decide which predictions are accepted.

        Args:
            uncertainty: The uncertainty criterion per instance, as a one-dimensional array.

        Returns:
            A boolean mask with the shape of ``uncertainty``, ``True`` where the prediction is accepted.

        Raises:
            TypeError: If ``uncertainty`` is not an array.
            ValueError: If ``uncertainty`` is not one-dimensional.
        """
        raise NotImplementedError


selector_registry = switch[SelectorName, type[Selector]]()
"""Registry of the selection rules by the :func:`~probly.selective_prediction.selector` keyword that names them."""


def _check_notion(notion: object) -> type[Notion] | Callable[[Any], Any]:
    """Resolve a notion name to its class and check the notion."""
    if isinstance(notion, str):
        try:
            return notion_registry[cast("NotionName", notion)]
        except KeyError:
            names = ", ".join(repr(name) for name in notion_registry)
            msg = f"notion must be one of {names}, got {notion!r}."
            raise ValueError(msg) from None
    if isinstance(notion, type):
        if not issubclass(notion, Notion):
            msg = f"notion must be a subclass of Notion if it is a class, got {notion.__name__}."
            raise TypeError(msg)
        return notion
    if not callable(notion):
        msg = f"notion must be a string, a notion class, or a callable, got {type(notion).__name__}."
        raise TypeError(msg)
    return notion


def _check_per_instance[U](values: U, name: str = "uncertainty") -> U:
    """Check that a per-instance array is one-dimensional."""
    ndim = getattr(values, "ndim", None)
    if ndim is None:
        msg = f"{name} must be an array with one value per instance, got {type(values).__name__}."
        raise TypeError(msg)
    if ndim != 1:
        shape = tuple(cast("Any", values).shape)
        msg = f"{name} must be one-dimensional with one value per instance, got shape {shape}."
        raise ValueError(msg)
    return values


@flexdispatch
def _to_float64_numpy(uncertainty: ArrayLikeWithoutSize) -> np.ndarray:
    """Copy a criterion array into a float64 NumPy array."""
    return np.asarray(uncertainty, dtype=np.float64)


def _calibration_array(values: ArrayLikeWithoutSize, name: str) -> np.ndarray:
    """Copy per-instance calibration data to float64 NumPy and check that it is usable."""
    array = _to_float64_numpy(_check_per_instance(values, name))
    if array.shape[0] == 0:
        msg = f"{name} must contain at least one calibration instance."
        raise ValueError(msg)
    if np.isnan(array).any():
        msg = f"{name} must not contain NaN."
        raise ValueError(msg)
    return array


@selector_registry.register("threshold")
class ThresholdSelector(Selector):
    """Selector with a fixed threshold, built by ``selector(threshold=...)``."""

    threshold: float

    def __init__(self, threshold: float) -> None:
        threshold = float(threshold)
        if math.isnan(threshold):
            msg = "threshold must not be NaN."
            raise ValueError(msg)
        self.threshold = threshold

    @override
    def __repr__(self) -> str:
        return f"selector(threshold={self.threshold})"

    @override
    def select(self, uncertainty: ArrayLikeWithoutSize) -> ArrayLikeWithoutSize:
        return cast("Any", _check_per_instance(uncertainty)) <= self.threshold


@selector_registry.register("coverage")
class CoverageSelector(Selector):
    """Selector for a target coverage with a split-conformal threshold, built by ``selector(coverage=...)``."""

    coverage: float
    threshold: float | None

    def __init__(self, coverage: float) -> None:
        coverage = float(coverage)
        if not 0.0 < coverage <= 1.0:
            msg = f"coverage must be in (0, 1], got {coverage}."
            raise ValueError(msg)
        self.coverage = coverage
        self.threshold = None

    @override
    def __repr__(self) -> str:
        return f"selector(coverage={self.coverage})"

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has been called."""
        return self.threshold is not None

    @override
    def select(self, uncertainty: ArrayLikeWithoutSize) -> ArrayLikeWithoutSize:
        if not self.is_calibrated:
            msg = f"{self!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)
        return cast("Any", _check_per_instance(uncertainty)) <= self.threshold

    def calibrate(self, uncertainty: ArrayLikeWithoutSize) -> Self:
        """Fit the threshold on the criterion of calibration instances, a one-dimensional array without NaN.

        Raises:
            ValueError: If there are fewer than ``coverage / (1 - coverage)`` calibration instances. The selector is
                left unchanged.
        """
        scores = _calibration_array(uncertainty, "uncertainty")
        n = scores.shape[0]
        # The tolerance keeps float error in (n + 1) * coverage from raising the rank (for up to ~9 significant digits).
        rank = math.ceil((n + 1) * self.coverage - 1e-9)
        if rank > n:
            if self.coverage < 1.0:
                needed = math.ceil(self.coverage / (1.0 - self.coverage) - 1e-9)
                msg = (
                    f"{n} calibration instances are too few for coverage={self.coverage}: at least {needed} are "
                    "needed. Use more calibration instances or a lower coverage."
                )
                raise ValueError(msg)
            self.threshold = math.inf
        else:
            self.threshold = float(np.partition(scores, rank - 1)[rank - 1])
        return self


def _binomial_upper_bound(errors: float, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper limit (Geifman and El-Yaniv, 2017, Lemma 3.1).

    The largest error rate under which at most ``errors`` errors in ``n`` trials have probability at least ``delta``.
    """
    if errors >= n:
        return 1.0
    return float(betaincinv(errors + 1, n - errors, 1.0 - delta))


@selector_registry.register("risk")
class SGRSelector(Selector):
    """Selector with a risk bound fitted by SGR on labeled data, built by ``selector(risk=..., delta=...)``."""

    takes_losses: ClassVar[bool] = True
    risk: float
    delta: float
    threshold: float | None
    bound: float | None

    def __init__(self, risk: float, delta: float) -> None:
        risk = float(risk)
        delta = float(delta)
        if not 0.0 < risk < 1.0:
            msg = f"risk must be in (0, 1), got {risk}."
            raise ValueError(msg)
        if not 0.0 < delta < 1.0:
            msg = f"delta must be in (0, 1), got {delta}."
            raise ValueError(msg)
        self.risk = risk
        self.delta = delta
        self.threshold = None
        self.bound = None

    @override
    def __repr__(self) -> str:
        return f"selector(risk={self.risk}, delta={self.delta})"

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has fitted a threshold."""
        return self.threshold is not None

    @override
    def select(self, uncertainty: ArrayLikeWithoutSize) -> ArrayLikeWithoutSize:
        if not self.is_calibrated:
            msg = f"{self!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)
        return cast("Any", _check_per_instance(uncertainty)) <= self.threshold

    def calibrate(self, uncertainty: ArrayLikeWithoutSize, losses: ArrayLikeWithoutSize) -> Self:
        """Fit the threshold on the criterion and the zero-one losses (0 or 1) of calibration instances.

        Both are one-dimensional arrays of the same length without NaN.

        Raises:
            ValueError: If no threshold can be certified. The selector is left unchanged.
        """
        scores = _calibration_array(uncertainty, "uncertainty")
        errors = _calibration_array(losses, "losses")
        n = scores.shape[0]
        if errors.shape[0] != n:
            msg = f"uncertainty and losses must have the same length, got {n} and {errors.shape[0]}."
            raise ValueError(msg)
        if not np.isin(errors, (0.0, 1.0)).all():
            msg = (
                "losses must be binary (0 or 1): the risk bound is the Clopper-Pearson limit for a Bernoulli loss. "
                "Use the zero-one loss."
            )
            raise ValueError(msg)

        order = np.argsort(scores, kind="stable")
        ascending = scores[order]
        cumulative_errors = np.concatenate([[0.0], np.cumsum(errors[order])])
        steps = (n - 1).bit_length()  # ceil(log2(n)), exact for every n
        level = self.delta / steps if steps > 0 else self.delta
        z_min, z_max = 1, n
        certified: tuple[float, float] | None = None
        for _ in range(steps):
            z = (z_min + z_max + 1) // 2
            threshold = float(ascending[n - z])
            accepted = int(np.searchsorted(ascending, threshold, side="right"))
            bound = _binomial_upper_bound(float(cumulative_errors[accepted]), accepted, level)
            if bound < self.risk:
                z_max = z
                certified = (threshold, bound)
            else:
                z_min = z
        if certified is None:
            msg = (
                f"No threshold could be certified for risk={self.risk} and delta={self.delta} with {n} calibration "
                "instances. Use more calibration data or a larger risk."
            )
            raise ValueError(msg)
        self.threshold, self.bound = certified
        return self


def selector(
    *,
    threshold: float | None = None,
    coverage: float | None = None,
    risk: float | None = None,
    delta: float | None = None,
    notion: NotionKey | Callable[[Any], Any] = "total",
    **options: Any,  # noqa: ANN401
) -> Selector:
    """Build the selection rule for a :class:`~probly.selective_prediction.SelectivePredictor`.

    Pass exactly one rule: ``threshold``, ``coverage``, or ``risk`` with ``delta``. The selector also works on its own,
    e.g. ``selector(coverage=0.9).calibrate(scores).select(new_scores)``. Every rule accepts a prediction whose
    criterion is at most its threshold, so ties are accepted and NaN is rejected.

    **Threshold.** The threshold is fixed and on the scale of the criterion, e.g. entropy in nats under the log loss,
    or differential entropy, which can be negative, for Gaussian regression. On the default criterion for categorical
    representations, ``1 - max_k p_k`` in ``[0, 1 - 1/K]`` for ``K`` classes, it is the cost of abstaining when an
    error costs 1: Chow's rule :cite:`chowOptimumRecognition1970` abstains where the model's own probability of an
    error exceeds this cost, which minimizes the expected cost if the probabilities are calibrated, e.g. by
    :func:`~probly.method.calibration.temperature_scaling`. It gives no guarantee.

    **Coverage.** Split conformal prediction on the criterion :cite:`angelopoulosGentleIntroduction2021`, a conformal
    version of the post-hoc step of SelectiveNet :cite:`geifmanSelectiveNet2019`: the threshold is the
    ``ceil((n + 1) * coverage)``-th smallest criterion of ``n`` calibration instances, which need no labels. If
    calibration and test instances are exchangeable, a new instance is accepted with probability at least
    ``coverage``, and at most about ``coverage + 1 / (n + 1)`` without ties; on a criterion with many ties, e.g. the
    zero-one epistemic part, the coverage can be well above the target. The risk is not controlled. With fewer than
    ``coverage / (1 - coverage)`` calibration instances, calibration raises a ``ValueError``.

    **Risk.** SGR (selection with guaranteed risk) :cite:`geifmanSelectiveClassification2017` fits the threshold on
    i.i.d. labeled calibration data so that, with probability at least ``1 - delta``, the selective risk
    ``E[loss * accepted] / E[accepted]`` is at most ``risk``. The coverage is not controlled and can be low. The loss
    must be the zero-one loss, since the bound is the Clopper-Pearson limit for a Bernoulli parameter. A binary search
    over the sorted calibration criteria, starting at the median, tests ``ceil(log2(n))`` thresholds at level
    ``delta / ceil(log2(n))`` each, never the largest criterion. It returns the last certified threshold, not the last
    tested one as the paper does; if none is certified, calibration raises a ``ValueError``. Errors among the most
    confident predictions cost power, since the search cannot recover from a failed test at the median. The tested
    thresholds depend on the labels seen in earlier steps, which the paper's union bound treats as fixed, so the
    guarantee is not rigorous in general.

    The selector also holds the notion of uncertainty that
    :class:`~probly.selective_prediction.SelectivePredictor` computes the criterion from; a selector applied to
    criteria computed by hand ignores it.

    The fitted rules work on a float64 NumPy copy of the criterion, so the threshold is the same on every backend and
    precision; gradients are not tracked.

    Args:
        threshold: Maximum criterion at which a prediction is accepted. ``inf`` accepts and ``-inf`` rejects every
            prediction.
        coverage: Target fraction of accepted predictions, in ``(0, 1]``.
        risk: Target selective risk, in ``(0, 1)``. Needs ``delta``.
        delta: Probability with which the bound on the risk may fail, in ``(0, 1)``. Needs ``risk``.
        notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"`` or ``"epistemic"``, or a
            function mapping the representation to the criterion per instance, higher meaning more uncertain. See
            :func:`~probly.selective_prediction.selective_predictor` for how it is computed.
        **options: Keywords of a rule registered in
            :data:`~probly.selective_prediction.selectors.selector_registry`, passed on to its class.

    Returns:
        A :class:`~probly.selective_prediction.selectors.Selector` whose ``threshold`` and ``bound`` show its state. The
        fitted rules add ``calibrate``, which fits in place and returns the selector: ``calibrate(uncertainty)`` for
        ``coverage``, and ``calibrate(uncertainty, losses)`` with the zero-one losses of the calibration predictions
        for ``risk``.

    Raises:
        TypeError: If not exactly one rule is given, ``risk`` and ``delta`` are not given together, or the rule does
            not take a keyword.
        ValueError: If a value is outside its range, ``threshold`` is NaN, or ``notion`` is a string that is not a
            notion of uncertainty.
    """
    checked_notion = _check_notion(notion)
    given = {
        name: value
        for name, value in (("threshold", threshold), ("coverage", coverage), ("risk", risk), ("delta", delta))
        if value is not None
    }
    given.update(options)
    if (risk is None) != (delta is None):
        msg = "risk and delta must be given together."
        raise TypeError(msg)
    rules = [rule for rule in selector_registry if rule in given]
    if len(rules) != 1:
        names = ", ".join(repr(rule) for rule in selector_registry)
        msg = f"Pass exactly one rule: {names}; risk also needs delta."
        raise TypeError(msg)
    rule = rules[0]
    rule_class = selector_registry[rule]
    try:
        built = rule_class(**given)
    except TypeError as error:
        accepted = set(inspect.signature(rule_class).parameters)
        unexpected = sorted(set(given) - accepted)
        if unexpected:
            msg = f"{unexpected[0]!r} is not an option of the {rule!r} rule."
            raise TypeError(msg) from None
        raise error from None
    built.notion = checked_notion
    return built
