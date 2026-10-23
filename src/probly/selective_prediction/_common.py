"""Selective predictors and the selectors they apply, see :mod:`probly.selective_prediction`."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Concatenate, Protocol, Self, cast, runtime_checkable

import numpy as np

from probly.calibrator import Calibrator
from probly.decider import Decider, categorical_from_mean
from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.predictor import Predictor, predict
from probly.quantification import (
    Decomposition,
    Notion,
    ScoringRule,
    ZeroOneLoss,
    notion_registry,
    quantify,
    scoring_rule_decomposition,
)
from probly.representation.distribution import CategoricalDistribution
from probly.representer import Representer

from .selectors._common import Selector, _check_per_instance, _to_float64_numpy

if TYPE_CHECKING:
    from collections.abc import Callable

    from ndlike.array_like import ArrayLikeWithoutSize

    from probly.quantification.notion import NotionKey, NotionName

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
    uncertainty: ArrayLikeWithoutSize
    accepted: ArrayLikeWithoutSize

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


def _check_selector_calibrated(predictor: SelectivePredictor[Any, Any]) -> None:
    """Raise before the forward pass if a fitted selector has not been calibrated."""
    selector = predictor.selector
    if not getattr(selector, "is_calibrated", True):
        msg = f"{selector!r} is not calibrated. Call calibrate() first."
        raise ValueError(msg)


def _call_user(function: Callable[..., Any], label: str, *arguments: Any) -> Any:  # noqa: ANN401
    """Call a user function and note on an exception what the function received."""
    try:
        return function(*arguments)
    except Exception as error:
        types = ", ".join(type(a).__name__ for a in arguments)
        if label == "loss":
            note = f"loss receives the decision of the decider and the labels ({types})"
        else:
            note = f"{label} receives the representation ({types}), not the model output"
        error.add_note(f"selective_predictor: {note}.")
        raise


def _decide(predictor: SelectivePredictor[Any, Any], representation: Any) -> Any:  # noqa: ANN401
    return _call_user(predictor.decider, "decider", representation)


def _losses(predictor: SelectivePredictor[Any, Any], representation: Any, y: Any) -> np.ndarray:  # noqa: ANN401
    """Task loss of the decision for the true label, per instance."""
    decision = _decide(predictor, representation)
    loss = predictor.loss
    if not isinstance(loss, ScoringRule):
        values = _to_float64_numpy(_call_user(cast("Callable[..., Any]", loss), "loss", decision, y))
    else:
        if not isinstance(decision, CategoricalDistribution):
            msg = (
                f"A scoring rule needs a categorical decision, got {type(decision).__name__}. "
                "Pass loss as a function of (decision, y), e.g. for regression."
            )
            raise TypeError(msg)
        _check_labels(decision, y)
        values = _to_float64_numpy(_call_user(loss, "loss", decision, y))
    if values.ndim != 1:
        msg = f"loss must return one value per instance, got shape {values.shape}."
        raise ValueError(msg)
    return values


def _check_labels(decision: CategoricalDistribution, y: ArrayLikeWithoutSize) -> None:
    """Check that the labels are one class index per instance, before a scoring rule picks the losses."""
    probabilities = _to_float64_numpy(decision.probabilities)
    num_instances, num_classes = probabilities.shape[0], probabilities.shape[-1]
    labels = _to_float64_numpy(y)
    if labels.ndim != 1 or labels.shape[0] != num_instances:
        msg = (
            "The labels must be a one-dimensional array with one class index per instance, got shape "
            f"{labels.shape} for {num_instances} instances."
        )
        raise ValueError(msg)
    if not (np.isfinite(labels).all() and (labels == np.round(labels)).all()):
        msg = "The labels must be integer class indices."
        raise ValueError(msg)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        msg = f"The labels must be in [0, {num_classes}), got values in [{labels.min()}, {labels.max()}]."
        raise ValueError(msg)


def _criterion(predictor: SelectivePredictor[Any, Any], representation: Any) -> ArrayLikeWithoutSize:  # noqa: ANN401
    notion = predictor.notion
    if not isinstance(notion, type):
        criterion = _call_user(notion, "notion", representation)
    else:
        criterion = _named_criterion(predictor, cast("type[Notion]", notion), representation)
    try:
        return _check_per_instance(criterion, "The criterion")
    except ValueError as error:
        error.add_note(
            "For a classifier, build the representer from a model with a predictor_type, e.g. "
            "representer(cast(model, predictor_type='logit_classifier')). "
            "For regression, pass notion as a function that reduces the criterion, e.g. "
            "lambda s: s.sample_var().squeeze(-1) for a one-output regression model."
        )
        raise


def _named_criterion(
    predictor: SelectivePredictor[Any, Any],
    notion: type[Notion],
    representation: Any,  # noqa: ANN401
) -> ArrayLikeWithoutSize:
    decomposition = _decomposition(predictor, representation)
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


def _decomposition(predictor: SelectivePredictor[Any, Any], representation: Any) -> Any:  # noqa: ANN401
    loss = predictor.loss
    if loss is None:
        return quantify(representation)
    # The default and a function loss fall back to quantify where the zero-one loss does not apply.
    fallback = loss is _ZERO_ONE_LOSS or not isinstance(loss, ScoringRule)
    if not isinstance(loss, ScoringRule):
        loss = _ZERO_ONE_LOSS
    registered = scoring_rule_decomposition.dispatch(
        type(representation), registry_meta_lookup=representation
    ) is not scoring_rule_decomposition.dispatch(object)
    if registered:
        return scoring_rule_decomposition(representation, loss)
    if fallback:
        return quantify(representation)
    msg = (
        f"{type(loss).__name__} is not supported for {type(representation).__name__}; "
        "omit loss to use the default decomposition of this representation, or pass loss=None."
    )
    raise NotImplementedError(msg)


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class SelectivePredictor[**In, D](
    Predictor[In, SelectivePrediction[D]],
    Calibrator[Concatenate[Any, In], Any],
    Protocol,
):
    """Predictor that answers or abstains per instance, built by :func:`selective_predictor`.

    Anything registered with ``SelectivePredictor.register`` that has these attributes predicts through the
    selective prediction pipeline of :func:`~probly.predictor.predict`.

    Attributes:
        selector: The selection rule, or ``None``. Its ``threshold`` and ``bound`` show its state.
        representer: The :class:`~probly.representer.Representer` that maps the inputs to the representation.
        notion: The notion class or the function that gives the criterion.
        loss: The task loss: a scoring rule, a function, or ``None``.
        decider: The :class:`~probly.decider.Decider` mapping a representation to the decision.
    """

    selector: Selector | None
    representer: Representer[Any, In, Any, Any]
    notion: type[Notion] | Callable[[Any], Any]
    loss: ScoringRule | Callable[[Any, Any], Any] | None
    decider: Decider[Any, Any]

    def calibrate(self, y_calib: Any, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:  # noqa: ANN401
        """Fit the selector on calibration data."""
        ...


@SelectivePredictor.register
class _SelectivePredictorBase[**In](ABC):
    """Selective predictor wrapping a representer, see :func:`selective_predictor`."""

    selector: Selector | None
    representer: Representer[Any, In, Any, Any]
    notion: type[Notion] | Callable[[Any], Any]
    loss: ScoringRule | Callable[[Any, Any], Any] | None
    decider: Decider[Any, Any]

    def __init__(
        self,
        base: Representer[Any, In, Any, Any],
        *,
        selector: Selector | None = None,
        notion: NotionKey | Callable[[Any], Any] = "total",
        loss: ScoringRule | Callable[[Any, Any], Any] | None = _ZERO_ONE_LOSS,
        decider: Decider[Any, Any] | None = None,
    ) -> None:
        super().__init__()
        if selector is not None and not isinstance(selector, Selector):
            msg = f"selector must be a Selector or None, got {type(selector).__name__}. Build one with selector(...)."
            raise TypeError(msg)
        if not isinstance(base, Representer):
            msg = (
                f"base must be a Representer, got {type(base).__name__}. Build one with "
                "probly.representer.representer(model, ...); cast a plain classifier first with "
                "probly.method.cast(model, predictor_type=...)."
            )
            raise TypeError(msg)
        self.selector = selector
        self.representer = base
        self.notion = _check_notion(notion)
        self.loss = _check_loss(loss, selector)
        self.decider = categorical_from_mean if decider is None else decider

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Predict and decide per instance whether to abstain, as :func:`~probly.predictor.predict` does.

        Args:
            *args: Inputs of the representer.
            **kwargs: Keyword arguments of the representer.

        Returns:
            The decision, criterion and acceptance mask per instance.

        Raises:
            TypeError: If there is no selector, or :func:`~probly.quantification.quantify` returns no decomposition.
            KeyError: If the decomposition lacks the component named by ``notion``.
            NotImplementedError: If a scoring rule does not apply to the representation.
            ValueError: If the selector is not calibrated or the criterion is not one-dimensional.
        """
        return predict(cast("Any", self), *args, **kwargs)

    def calibrate(self, y_calib: Any, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:  # noqa: ANN401
        """Fit the selector on calibration data, with the criterion computed as in :meth:`predict`.

        The labels come first, as for the other calibrators of probly.

        Args:
            y_calib: ``None`` for a selector that takes no losses, e.g. ``selector(coverage=...)``. For one that does,
                e.g. ``selector(risk=..., delta=...)``, the labels to compute the losses of the decisions: one class
                index per instance for a scoring rule, anything the function accepts for a function loss.
            *calib_args: Inputs of the representer, as in :meth:`predict`.
            **calib_kwargs: Keyword arguments of the representer, as in :meth:`predict`.

        Returns:
            The selective predictor itself.

        Raises:
            TypeError: If there is no selector to fit (none or a fixed threshold) or no input of the representer, if
                the labels do not fit the selector, or if a scoring rule meets a decision that is not categorical.
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
                "No input of the representer was given. Pass the labels first and then the inputs, e.g. "
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
        uncertainty = _criterion(self, representation)
        if y_calib is None:
            calibrate(uncertainty)
        else:
            calibrate(uncertainty, _losses(self, representation, y_calib))
        return self

    def evaluate(self, y: Any, *args: Any, metrics: str | list[str] | None = None, **kwargs: Any) -> dict[str, float]:  # noqa: ANN401
        """Evaluate the criterion, and the selector if there is one, on labeled data in one forward pass.

        The metrics of :func:`~probly.evaluation.selective_prediction.evaluate_selective_prediction` are computed from
        the losses of the decisions and the criterion, with NaN mapped to ``+inf`` as the selectors reject it. A
        selector adds ``"selected:coverage"``, the fraction of accepted instances, and ``"selected:risk"``, their mean
        loss (NaN if none is accepted).

        Args:
            y: The labels, as in :meth:`calibrate`.
            *args: Inputs of the representer, as in :meth:`predict`.
            metrics: The metrics of ``evaluate_selective_prediction``; reserved, so not passed on to the representer.
            **kwargs: Keyword arguments of the representer, as in :meth:`predict`.

        Returns:
            A dictionary of floats.

        Raises:
            TypeError: If ``loss`` is ``None`` or no input of the representer is given.
            ValueError: If a fitted selector is not calibrated, or the criterion or losses are invalid. Unlike
                :meth:`predict`, the metrics need a criterion without ``-inf`` and finite losses, which e.g. a
                :class:`~probly.quantification.LogLoss` breaks if the probability of the label underflows in float32.
        """
        if self.loss is None:
            msg = "evaluate needs losses. Pass a scoring rule or a function loss."
            raise TypeError(msg)
        if not args and not kwargs:
            msg = (
                "No input of the representer was given. Pass the labels first and then the inputs, e.g. evaluate(y, x)."
            )
            raise TypeError(msg)
        selector = self.selector
        _check_selector_calibrated(self)
        representation = self.representer.represent(*args, **kwargs)
        native_criterion = _criterion(self, representation)
        criterion = _to_float64_numpy(native_criterion)
        losses = _losses(self, representation, y)
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


@predict.register(SelectivePredictor)
def predict_selective_prediction[**In](
    predictor: SelectivePredictor[In, Any], *args: In.args, **kwargs: In.kwargs
) -> SelectivePrediction[Any]:
    """Predict and decide per instance whether to abstain; the arguments go to the representer."""
    selector = predictor.selector
    if selector is None:
        msg = "No selector to select with. Pass selector=selector(...), or use evaluate."
        raise TypeError(msg)
    _check_selector_calibrated(predictor)
    representation = predictor.representer.represent(*args, **kwargs)
    uncertainty = _criterion(predictor, representation)
    return SelectivePrediction(
        decision=_decide(predictor, representation),
        uncertainty=uncertainty,
        accepted=selector.select(uncertainty),
    )


@SelectivePredictor.register_factory
def selective_predictor[**In](
    base: Representer[Any, In, Any, Any],
    *,
    selector: Selector | None = None,
    notion: NotionKey | Callable[[Any], Any] = "total",
    loss: ScoringRule | Callable[[Any, Any], Any] | None = _ZERO_ONE_LOSS,
    decider: Decider[Any, Any] | None = None,
) -> SelectivePredictor[In, Any]:
    r"""Selective predictor that answers or abstains based on an uncertainty criterion.

    The selection rule is a :class:`~probly.selective_prediction.selectors.Selector` built with
    :func:`~probly.selective_prediction.selector`; rules fitted on data are fitted with ``calibrate``. Without a
    selector, predicting raises and ``evaluate`` reports only the ranking metrics.

    The representer is built with :func:`~probly.representer.representer` from a model probly can represent, e.g. an
    ensemble, an MC-dropout model, an evidential model, an object with ``predict_proba``, or a plain classifier declared
    with :func:`~probly.method.cast`; options such as ``num_samples`` are set on it, e.g.
    ``representer(model, num_samples=100)``. The criterion and the decision come from the same representation. For
    other models, apply a :func:`~probly.selective_prediction.selector` to a criterion computed by hand.

    The criterion is the component of the uncertainty decomposition selected by ``notion``: the total uncertainty by
    default, recommended for selective prediction :cite:`hofmanAdjustingMeasures2025`, or its aleatoric or epistemic
    part; the epistemic part suits the rejection of out-of-distribution instances. For samples of categorical
    distributions, the task loss ``loss`` induces the decomposition, see
    :func:`~probly.quantification.decomposition.scoring_rule.scoring_rule_decomposition`. With ``theta_bar``
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
    :func:`~ndlike.jax_functions.jax_concatenate`), and pass the representer of the identity,
    ``representer(cast(lambda r: r, predictor_type=RepresentationPredictor))``. With PyTorch, wrap ``predict``,
    ``calibrate`` and ``evaluate`` in ``torch.no_grad()``, as they do not stop gradients.

    Args:
        base: The :class:`~probly.representer.Representer` that maps the inputs to the representation, built with
            :func:`~probly.representer.representer`.
        selector: The selection rule, built with :func:`~probly.selective_prediction.selector`, or ``None``.
            ``calibrate`` fits it in place, so predictors sharing a selector share its state.
        notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"`` or ``"epistemic"``, or a
            function mapping the representation to the criterion per instance, higher meaning more uncertain.
        loss: The task loss: a scoring rule for integer labels (default: zero-one loss), a function
            ``loss(decision, y)`` returning one loss per instance, e.g. for regression, or ``None`` for
            :func:`~probly.quantification.quantify` everywhere and no losses.
        decider: A :class:`~probly.decider.Decider` mapping the representation to the decision. Defaults to
            :func:`~probly.decider.categorical_from_mean`.

    Returns:
        The selective predictor, with ``predict``, ``calibrate`` and ``evaluate``.

    Raises:
        TypeError: If ``selector`` is not a :class:`~probly.selective_prediction.selectors.Selector` or ``None``,
            ``base`` is not a :class:`~probly.representer.Representer`, or a selector that needs losses meets
            ``loss=None``.
        ValueError: If ``notion`` is a string that is not a notion of uncertainty.
    """
    return _SelectivePredictorBase(
        base,
        selector=selector,
        notion=notion,
        loss=loss,
        decider=decider,
    )


__all__ = ["SelectivePrediction", "SelectivePredictor", "selective_predictor"]
