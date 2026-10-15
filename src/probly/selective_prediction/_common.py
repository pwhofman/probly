"""Selective predictors and the selectors they apply, see :mod:`probly.selective_prediction`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, final, override

from flextype import flexdispatch
import numpy as np
from scipy.special import betaincinv

from probly.decider import categorical_from_mean
from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.quantification import (
    Decomposition,
    Notion,
    ScoringRule,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    ZeroOneLoss,
    notion_registry,
    quantify,
)
from probly.quantification.decomposition.decomposition import ConstantTotalDecomposition
from probly.quantification.measure.distribution import generalized_entropy_of_expected
from probly.representation.distribution import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
)
from probly.representation.sample import create_sample
from probly.representer import representer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from probly.predictor import Predictor
    from probly.quantification.notion import NotionKey, NotionName
    from probly.representer import Representer

# Default loss, told from an explicit ZeroOneLoss() by identity, because only it falls back to quantify.
_ZERO_ONE_LOSS = ZeroOneLoss()


@dataclass(frozen=True, slots=True)
class SelectivePrediction[D]:
    """Output of a selective predictor.

    Attributes:
        decision: The decision per instance, also where the predictor abstains.
        uncertainty: The uncertainty criterion per instance; higher means more uncertain.
        accepted: Boolean mask per instance, ``False`` where the predictor abstains.
    """

    decision: D
    uncertainty: Any
    accepted: Any

    @property
    def coverage(self) -> float:
        """Fraction of accepted predictions, or NaN for an empty batch.

        As a Python float, it cannot be used under ``jax.jit``; take the mean of :attr:`accepted` there.
        """
        accepted: Any = self.accepted
        num_instances = math.prod(accepted.shape)
        if num_instances == 0:
            return math.nan
        return float(accepted.sum()) / num_instances


class Selector(ABC):
    """Base class for rules that accept or reject predictions based only on their uncertainty criterion.

    A selector does not see the model, so it also works on scores computed for any other model. The criterion is a
    one-dimensional array with one value per instance. As in
    :obj:`~probly.metrics.selective_prediction.risk_coverage_curve`, higher values mean more uncertain and are
    rejected first, so a confidence score has to be negated. A NaN criterion is rejected.

    A subclass overrides :meth:`select`. A rule fitted on data implements :class:`~probly.calibrator.Calibrator` on
    criterion arrays, sets :attr:`takes_losses` if ``calibrate`` also takes the losses of the calibration predictions,
    and has an ``is_calibrated`` attribute, which :class:`SelectivePredictor` checks before the forward pass; while it
    is ``False``, :meth:`select` raises a ``ValueError``.

    Attributes:
        takes_losses: Whether ``calibrate`` takes the losses of the calibration predictions as its second argument.
        threshold: The fixed or fitted threshold on the criterion, or ``None`` if there is none (yet).
        bound: The certified bound on the risk, or ``None`` if there is none (yet).
    """

    takes_losses: ClassVar[bool] = False
    threshold: float | None = None
    bound: float | None = None

    @abstractmethod
    def select(self, uncertainty: Any) -> Any:  # noqa: ANN401
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
def _to_float64_numpy(uncertainty: Any) -> np.ndarray:  # noqa: ANN401
    """Copy a criterion array into a float64 NumPy array."""
    return np.asarray(uncertainty, dtype=np.float64)


def _calibration_array(values: Any, name: str) -> np.ndarray:  # noqa: ANN401
    """Copy per-instance calibration data to float64 NumPy and check that it is usable."""
    array = _to_float64_numpy(_check_per_instance(values, name))
    if array.shape[0] == 0:
        msg = f"{name} must contain at least one calibration instance."
        raise ValueError(msg)
    if np.isnan(array).any():
        msg = f"{name} must not contain NaN."
        raise ValueError(msg)
    return array


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
    def select(self, uncertainty: Any) -> Any:
        return _check_per_instance(uncertainty) <= self.threshold


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
    def select(self, uncertainty: Any) -> Any:
        if not self.is_calibrated:
            msg = f"{self!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)
        return _check_per_instance(uncertainty) <= self.threshold

    def calibrate(self, uncertainty: Any) -> Self:  # noqa: ANN401
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
    def select(self, uncertainty: Any) -> Any:
        if not self.is_calibrated:
            msg = f"{self!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)
        return _check_per_instance(uncertainty) <= self.threshold

    def calibrate(self, uncertainty: Any, losses: Any) -> Self:  # noqa: ANN401
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
) -> Selector:
    """Build the selection rule for a :class:`SelectivePredictor` from the keywords that name its guarantee.

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

    The fitted rules work on a float64 NumPy copy of the criterion, so the threshold is the same on every backend and
    precision; gradients are not tracked.

    Args:
        threshold: Maximum criterion at which a prediction is accepted. ``inf`` accepts and ``-inf`` rejects every
            prediction.
        coverage: Target fraction of accepted predictions, in ``(0, 1]``.
        risk: Target selective risk, in ``(0, 1)``. Needs ``delta``.
        delta: Probability with which the bound on the risk may fail, in ``(0, 1)``. Needs ``risk``.

    Returns:
        A :class:`Selector` whose ``threshold`` and ``bound`` show its state. The fitted rules add ``calibrate``, which
        fits in place and returns the selector: ``calibrate(uncertainty)`` for ``coverage``, and
        ``calibrate(uncertainty, losses)`` with the zero-one losses of the calibration predictions for ``risk``.

    Raises:
        TypeError: If not exactly one rule is given, or ``risk`` and ``delta`` are not given together.
        ValueError: If a value is outside its range or ``threshold`` is NaN.
    """
    if (risk is None) != (delta is None):
        msg = "risk and delta must be given together."
        raise TypeError(msg)
    if sum(value is not None for value in (threshold, coverage, risk)) != 1:
        msg = "Pass exactly one rule: threshold, coverage, or risk with delta."
        raise TypeError(msg)
    if threshold is not None:
        return ThresholdSelector(threshold)
    if coverage is not None:
        return CoverageSelector(coverage)
    return SGRSelector(cast("float", risk), cast("float", delta))


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


def _check_loss(loss: Any, selector: Selector | None) -> Any:  # noqa: ANN401
    """Check the loss and that it fits the selector."""
    if loss is not None and not isinstance(loss, ScoringRule) and not callable(loss):
        msg = f"loss must be a scoring rule, a function, or None, got {type(loss).__name__}."
        raise TypeError(msg)
    if selector is not None and selector.takes_losses and loss is None:
        msg = f"{selector!r} needs losses. Pass a task loss, e.g. ZeroOneLoss()."
        raise TypeError(msg)
    return loss


class SelectivePredictor[**In]:
    r"""Selective predictor that answers or abstains based on an uncertainty criterion.

    The selection rule is a :class:`Selector` built with :func:`selector`; rules fitted on data are fitted with
    :meth:`calibrate`. Without a selector, :meth:`predict` raises and :meth:`evaluate` reports only the ranking
    metrics.

    The model must be accepted by :func:`~probly.representer.representer`, e.g. an ensemble, an MC-dropout model, an
    evidential model, an object with ``predict_proba``, or a plain classifier declared with
    :func:`~probly.method.cast`. The criterion and the decision come from the same representation. For other models,
    apply a :func:`selector` to a criterion computed by hand.

    The criterion is the component of the uncertainty decomposition selected by ``notion``: the total uncertainty by
    default, recommended for selective prediction :cite:`hofmanAdjustingMeasures2025`, or its aleatoric or epistemic
    part; the epistemic part suits the rejection of out-of-distribution instances. For samples of categorical
    distributions, the task loss ``loss`` induces the decomposition, see
    :class:`~probly.quantification.decomposition.scoring_rule.SecondOrderScoringRuleDecomposition`. With ``theta_bar``
    the mean prediction, the default zero-one loss gives the total ``1 - max_k theta_bar_k``, the model's own
    probability that the default decision is wrong, the aleatoric part ``E[1 - max_k theta_k]``, and the epistemic
    part, their difference; the log loss gives the entropy and the Brier loss the Gini decomposition. Other
    representations:

    - a single categorical distribution: the expected loss of the best prediction, a total uncertainty only;
    - Dirichlet distributions: the zero-one decomposition; other scoring rules raise a ``NotImplementedError``;
    - any other representation, e.g. of a regression model or a credal set: :func:`~probly.quantification.quantify`
      if ``loss`` is the default or a function; an explicit scoring rule raises a ``NotImplementedError``.

    A decomposition that a method registers for its own representation (DARE, SNGP, heteroscedastic networks, the
    vacuity of evidential models) is used only with ``loss=None``. How the arguments are used:

    =================  ===============================  =====================  ================
    \                  Scoring rule                     Function loss          ``loss=None``
    =================  ===============================  =====================  ================
    criterion, name    decomposition of ``loss``        as the default loss    ``quantify``
    criterion, func    ``notion(rep)``                  ``notion(rep)``        ``notion(rep)``
    decision           ``decider(rep)``                 ``decider(rep)``       ``decider(rep)``
    losses             ``loss.loss(decision)[i, y_i]``  ``loss(decision, y)``  none
    =================  ===============================  =====================  ================

    Regression needs a function ``notion``, ``decider`` and ``loss``, e.g. for MC dropout
    ``notion=lambda s: s.sample_var().squeeze(-1)``, ``decider=lambda s: s.sample_mean().squeeze(-1)`` and
    ``loss=lambda d, y: (d - y) ** 2``. For large data, concatenate precomputed representations on the instance axis,
    keeping their type (``torch.cat``, ``np.concatenate`` or
    :func:`~probly.representation.jax_functions.jax_concatenate`), and wrap the identity
    ``cast(lambda r: r, predictor_type=RepresentationPredictor)``. With PyTorch, wrap :meth:`predict`,
    :meth:`calibrate` and :meth:`evaluate` in ``torch.no_grad()``, as they do not stop gradients.

    Attributes:
        model: The wrapped model.
        selector: The selection rule, or ``None``. Its ``threshold`` and ``bound`` show its state.
        representer: The representer of the model.
        notion: The notion class or the function that gives the criterion.
        loss: The task loss: a scoring rule, a function, or ``None``.
        decider: Function mapping a representation to the decision.
    """

    model: Predictor[In, Any]
    selector: Selector | None
    representer: Representer[Any, In, Any, Any]
    notion: type[Notion] | Callable[[Any], Any]
    loss: ScoringRule | Callable[[Any, Any], Any] | None
    decider: Callable[[Any], Any]

    def __init__(
        self,
        model: Predictor[In, Any],
        *,
        selector: Selector | None = None,
        notion: NotionKey | Callable[[Any], Any] = "total",
        loss: ScoringRule | Callable[[Any, Any], Any] | None = _ZERO_ONE_LOSS,
        decider: Callable[[Any], Any] | None = None,
        representer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the selective predictor.

        Args:
            model: A model accepted by :func:`~probly.representer.representer`.
            selector: The selection rule, built with :func:`selector`, or ``None``. :meth:`calibrate` fits it in place,
                so predictors sharing a selector share its state.
            notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"`` or ``"epistemic"``, or a
                function mapping the representation to the criterion per instance, higher meaning more uncertain.
            loss: The task loss: a scoring rule for integer labels (default: zero-one loss), a function
                ``loss(decision, y)`` returning one loss per instance, e.g. for regression, or ``None`` for
                :func:`~probly.quantification.quantify` everywhere and no losses.
            decider: Function mapping the representation to the decision. Defaults to
                :func:`~probly.decider.categorical_from_mean`.
            representer_kwargs: Keyword arguments for :func:`~probly.representer.representer`, e.g.
                ``{"num_samples": ...}``.

        Raises:
            TypeError: If ``selector`` is not a :class:`Selector` or ``None``, the model cannot be represented, or a
                selector that needs losses meets ``loss=None``.
            ValueError: If ``notion`` is a string that is not a notion of uncertainty.
        """
        if selector is not None and not isinstance(selector, Selector):
            msg = f"selector must be a Selector or None, got {type(selector).__name__}. Build one with selector(...)."
            raise TypeError(msg)
        if representer.dispatch(type(model), registry_meta_lookup=model) is representer.dispatch(object):
            msg = (
                f"{type(model).__name__} is not a model that probly can represent. Cast it with probly.method.cast, "
                "or compute the criterion yourself and apply a selector to it, see selector(...)."
            )
            raise TypeError(msg)
        self.model = model
        self.selector = selector
        self.notion = _check_notion(notion)
        self.loss = _check_loss(loss, selector)
        self.representer = representer(model, **(representer_kwargs or {}))
        self.decider = categorical_from_mean if decider is None else decider

    def _check_selector_calibrated(self) -> None:
        """Raise before the forward pass if a fitted selector has not been calibrated."""
        selector = self.selector
        if not getattr(selector, "is_calibrated", True):
            msg = f"{selector!r} is not calibrated. Call calibrate() first."
            raise ValueError(msg)

    def _call_user(self, function: Callable[..., Any], label: str, *arguments: Any) -> Any:  # noqa: ANN401
        """Call a user function and note on an exception what the function received."""
        try:
            return function(*arguments)
        except Exception as error:
            types = ", ".join(type(a).__name__ for a in arguments)
            if label == "loss":
                note = f"loss receives the decision of the decider and the labels ({types})"
            else:
                note = f"{label} receives the representation ({types}), not the model output"
            error.add_note(f"SelectivePredictor: {note}.")
            raise

    @final
    def predict(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Predict and decide per instance whether to abstain; the arguments go to the model's representer.

        Returns:
            The decision, criterion and acceptance mask per instance.

        Raises:
            TypeError: If there is no selector, or :func:`~probly.quantification.quantify` returns no decomposition.
            KeyError: If the decomposition lacks the component named by ``notion``.
            NotImplementedError: If a scoring rule does not apply to the representation.
            ValueError: If the selector is not calibrated or the criterion is not one-dimensional.
        """
        selector = self.selector
        if selector is None:
            msg = "No selector to select with. Pass selector=selector(...), or use evaluate."
            raise TypeError(msg)
        self._check_selector_calibrated()
        representation = self.representer.represent(*args, **kwargs)
        uncertainty = self._criterion(representation)
        return SelectivePrediction(
            decision=self._decide(representation),
            uncertainty=uncertainty,
            accepted=selector.select(uncertainty),
        )

    @final
    def calibrate(self, y_calib: Any, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:  # noqa: ANN401
        """Fit the selector on calibration data, with the criterion computed as in :meth:`predict`.

        The labels come first, as for the other calibrators of probly.

        Args:
            y_calib: ``None`` for a selector that takes no losses, e.g. ``selector(coverage=...)``. For one that does,
                e.g. ``selector(risk=..., delta=...)``, the labels to compute the losses of the decisions: one class
                index per instance for a scoring rule, anything the function accepts for a function loss.
            *calib_args: Inputs of the model, as in :meth:`predict`.
            **calib_kwargs: Keyword arguments of the model, as in :meth:`predict`.

        Returns:
            The selective predictor itself.

        Raises:
            TypeError: If there is no selector to fit (none or a fixed threshold) or no model input, if the labels do
                not fit the selector, or if a scoring rule meets a decision that is not categorical.
            ValueError: If the criterion or the labels are invalid, or the losses are not binary for the risk rule.
            NotImplementedError: If a scoring rule does not apply to the representation.
        """
        selector = self.selector
        if selector is None:
            msg = "There is no selector to calibrate. Pass selector=selector(...) with coverage, or risk and delta."
            raise TypeError(msg)
        calibrate = getattr(selector, "calibrate", None)
        if not callable(calibrate):
            msg = f"{selector!r} is not fitted on data, so there is nothing to calibrate."
            raise TypeError(msg)
        if not calib_args and not calib_kwargs:
            msg = (
                "No input of the model was given. Pass the labels first and then the inputs, e.g. "
                "calibrate(y_calib, x_calib), with y_calib=None for the coverage rule."
            )
            raise TypeError(msg)
        if selector.takes_losses and y_calib is None:
            msg = (
                f"{selector!r} needs the labels of the calibration instances. "
                "Pass them first: calibrate(y_calib, x_calib)."
            )
            raise TypeError(msg)
        if selector.takes_losses and self.loss is None:
            msg = f"{selector!r} needs losses, but loss is None. Set a task loss, e.g. ZeroOneLoss()."
            raise TypeError(msg)
        if not selector.takes_losses and y_calib is not None:
            msg = f"{selector!r} takes no labels. Pass None instead: calibrate(None, x_calib)."
            raise TypeError(msg)
        representation = self.representer.represent(*calib_args, **calib_kwargs)
        uncertainty = self._criterion(representation)
        if y_calib is None:
            calibrate(uncertainty)
        else:
            calibrate(uncertainty, self._losses(representation, y_calib))
        return self

    @final
    def evaluate(self, y: Any, *args: Any, metrics: str | list[str] | None = None, **kwargs: Any) -> dict[str, float]:  # noqa: ANN401
        """Evaluate the criterion, and the selector if there is one, on labeled data in one forward pass.

        The metrics of :func:`~probly.evaluation.selective_prediction.evaluate_selective_prediction` are computed from
        the losses of the decisions and the criterion, with NaN mapped to ``+inf`` as the selectors reject it. A
        selector adds ``"selected:coverage"``, the fraction of accepted instances, and ``"selected:risk"``, their mean
        loss (NaN if none is accepted).

        Args:
            y: The labels, as in :meth:`calibrate`.
            *args: Inputs of the model, as in :meth:`predict`.
            metrics: The metrics of ``evaluate_selective_prediction``; reserved, so not passed on to the model.
            **kwargs: Keyword arguments of the model, as in :meth:`predict`.

        Returns:
            A dictionary of floats.

        Raises:
            TypeError: If ``loss`` is ``None`` or no input of the model is given.
            ValueError: If a fitted selector is not calibrated, or the criterion or losses are invalid. Unlike
                :meth:`predict`, the metrics need a criterion without ``-inf`` and finite losses, which e.g. a
                :class:`~probly.quantification.LogLoss` breaks if the probability of the label underflows in float32.
        """
        if self.loss is None:
            msg = "evaluate needs losses. Pass a scoring rule or a function loss."
            raise TypeError(msg)
        if not args and not kwargs:
            msg = "No input of the model was given. Pass the labels first and then the inputs, e.g. evaluate(y, x)."
            raise TypeError(msg)
        selector = self.selector
        self._check_selector_calibrated()
        representation = self.representer.represent(*args, **kwargs)
        native_criterion = self._criterion(representation)
        criterion = _to_float64_numpy(native_criterion)
        losses = self._losses(representation, y)
        if criterion.shape != losses.shape:
            msg = f"criterion and losses must have the same shape, got {criterion.shape} and {losses.shape}."
            raise ValueError(msg)
        result = evaluate_selective_prediction(np.where(np.isnan(criterion), np.inf, criterion), losses, metrics)
        if selector is not None:
            # Select on the native criterion, as predict does, so that float32 is compared in float32 on both paths.
            accepted = _to_float64_numpy(selector.select(native_criterion)) > 0
            result["selected:coverage"] = float(accepted.mean()) if accepted.size else math.nan
            result["selected:risk"] = float(losses[accepted].mean()) if accepted.any() else math.nan
        return result

    def _decide(self, representation: Any) -> Any:  # noqa: ANN401
        return self._call_user(self.decider, "decider", representation)

    def _losses(self, representation: Any, y: Any) -> np.ndarray:  # noqa: ANN401
        """Task loss of the decision for the true label, per instance."""
        decision = self._decide(representation)
        loss = self.loss
        if not isinstance(loss, ScoringRule):
            values = _to_float64_numpy(self._call_user(cast("Callable[..., Any]", loss), "loss", decision, y))
            if values.ndim != 1:
                msg = f"loss must return one value per instance, got shape {values.shape}."
                raise ValueError(msg)
            return values
        if not isinstance(decision, CategoricalDistribution):
            msg = (
                f"A scoring rule needs a categorical decision, got {type(decision).__name__}. "
                "Pass loss as a function of (decision, y), e.g. for regression."
            )
            raise TypeError(msg)
        loss_matrix = _to_float64_numpy(loss.loss(decision))
        labels = _to_float64_numpy(y)
        if labels.ndim != 1 or loss_matrix.ndim != 2 or labels.shape[0] != loss_matrix.shape[0]:
            msg = (
                "The labels must be a one-dimensional array with one class index per instance, got shape "
                f"{labels.shape} for losses of shape {loss_matrix.shape}."
            )
            raise ValueError(msg)
        if not (np.isfinite(labels).all() and (labels == np.round(labels)).all()):
            msg = "The labels must be integer class indices."
            raise ValueError(msg)
        if labels.size and (labels.min() < 0 or labels.max() >= loss_matrix.shape[1]):
            msg = f"The labels must be in [0, {loss_matrix.shape[1]}), got values in [{labels.min()}, {labels.max()}]."
            raise ValueError(msg)
        return loss_matrix[np.arange(labels.shape[0]), labels.astype(np.intp)]

    def _criterion(self, representation: Any) -> Any:  # noqa: ANN401
        notion = self.notion
        if not isinstance(notion, type):
            criterion = self._call_user(notion, "notion", representation)
        else:
            criterion = self._named_criterion(cast("type[Notion]", notion), representation)
        try:
            return _check_per_instance(criterion, "The criterion")
        except ValueError as error:
            error.add_note(
                "For a classifier, transform the model with a predictor_type, e.g. predictor_type='logit_classifier'. "
                "For regression, pass notion as a function that reduces the criterion, e.g. "
                "lambda s: s.sample_var().squeeze(-1) for a one-output regression model."
            )
            raise

    def _named_criterion(self, notion: type[Notion], representation: Any) -> Any:  # noqa: ANN401
        decomposition = self._decomposition(representation)
        if not isinstance(decomposition, Decomposition):
            msg = f"Expected quantify to return a Decomposition, got {type(decomposition).__name__}."
            raise TypeError(msg)
        try:
            return decomposition[notion]
        except KeyError:
            available = ", ".join(component.__name__ for component in decomposition.components)
            msg = (
                f"{type(representation).__name__} has no {notion.__name__} in its decomposition, only: {available}. "
                "Pass a different notion."
            )
            raise KeyError(msg) from None

    def _decomposition(self, representation: Any) -> Any:  # noqa: ANN401
        loss = self.loss
        if loss is None:
            return quantify(representation)
        # The default and a function loss fall back to quantify where the zero-one loss does not apply.
        fallback = loss is _ZERO_ONE_LOSS or not isinstance(loss, ScoringRule)
        if not isinstance(loss, ScoringRule):
            loss = _ZERO_ONE_LOSS
        if isinstance(representation, CategoricalDistributionSample):
            if isinstance(loss, ZeroOneLoss):
                # Unlike total minus aleatoric, its epistemic part is exactly zero where all members agree.
                return SecondOrderZeroOneDecomposition(representation)
            return SecondOrderScoringRuleDecomposition(representation, loss)
        if isinstance(representation, CategoricalDistribution):
            # A sample with one member reuses the backend implementations.
            return ConstantTotalDecomposition(generalized_entropy_of_expected(create_sample([representation]), loss))
        if isinstance(representation, DirichletDistribution):
            if not isinstance(loss, ZeroOneLoss):
                msg = (
                    f"{type(loss).__name__} is not supported for Dirichlet distributions; "
                    "use ZeroOneLoss() or loss=None."
                )
                raise NotImplementedError(msg)
            return SecondOrderZeroOneDecomposition(representation)
        if fallback:
            return quantify(representation)
        msg = (
            f"{type(loss).__name__} is not supported for {type(representation).__name__}; "
            "omit loss to use the default decomposition of this representation, or pass loss=None."
        )
        raise NotImplementedError(msg)
