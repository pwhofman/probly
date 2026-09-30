"""Selective predictors that answer or abstain based on an uncertainty criterion.

A selective predictor is allowed to abstain: for each instance, it either returns the prediction of the wrapped model
or refrains from predicting. The choice is made by a rule on an uncertainty criterion computed from the model's
representation, which is why the wrapped model has to be uncertainty-aware.

The module separates the selection rule from the pipeline that computes the criterion. A :class:`Selector` decides
on arrays of uncertainty values alone and can therefore be used with any model or with precomputed scores.
:class:`SelectivePredictor` wraps a model transformed by probly, computes the criterion and the decision from its
representation, and delegates the selection to a :class:`Selector`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, cast, final, override

from probly.decider import categorical_from_mean
from probly.quantification import Decomposition, notion_registry, quantify
from probly.representer import representer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from probly.predictor import Predictor
    from probly.quantification import Notion
    from probly.quantification.notion import NotionKey, NotionName
    from probly.representation import Representation
    from probly.representer import Representer


@dataclass(frozen=True, slots=True)
class SelectivePrediction[D]:
    """Output of a selective predictor.

    The decision is returned for every instance, including those the predictor abstains on, so that rejected
    predictions remain available for inspection and evaluation.

    Attributes:
        decision: The decision per instance, e.g. a categorical distribution whose argmax is the predicted class.
        uncertainty: The uncertainty criterion per instance, where higher means more uncertain.
        accepted: Boolean mask per instance; ``True`` means the prediction is accepted, ``False`` means the
            predictor abstains.
    """

    decision: D
    uncertainty: Any
    accepted: Any

    @property
    def coverage(self) -> float:
        """Fraction of instances whose prediction is accepted."""
        accepted: Any = self.accepted
        return float(accepted.sum()) / math.prod(accepted.shape)


class Selector(ABC):
    """Base class for selection rules.

    A selector decides per instance whether a prediction is accepted, based only on its uncertainty criterion. It does
    not know the model the criterion comes from, so it works with the criterion computed by a
    :class:`SelectivePredictor` as well as with scores computed by the user for any other model. The criterion follows
    the convention of :func:`~probly.evaluation.selective_prediction.selective_prediction`: higher values mean more
    uncertain, so the instances with the largest criterion are rejected first. A confidence score has to be negated.

    Types of selective prediction differ in their selector: each is a subclass that overrides :meth:`select`.
    Selectors whose rule has to be fitted on data (e.g. a threshold chosen for a target coverage or risk) should
    implement the :class:`~probly.calibrator.Calibrator` protocol on arrays of the criterion and raise a
    ``ValueError`` from :meth:`select` while they are not calibrated.
    """

    @abstractmethod
    def select(self, uncertainty: Any) -> Any:  # noqa: ANN401
        """Decide which predictions are accepted.

        Args:
            uncertainty: The uncertainty criterion per instance.

        Returns:
            A boolean array with the batch shape of ``uncertainty``; ``True`` means the prediction is accepted.
        """
        raise NotImplementedError

    def __call__(self, uncertainty: Any) -> Any:  # noqa: ANN401
        """Alias for :meth:`select`."""
        return self.select(uncertainty)


class ThresholdSelector(Selector):
    """Selector that accepts a prediction if its uncertainty does not exceed a fixed threshold.

    A prediction is accepted if and only if its uncertainty criterion is less than or equal to the threshold, so
    ties at the threshold are accepted and NaN criteria are rejected. The threshold lives on the scale of the
    criterion, e.g. entropy in nats for classification, where ``log(K)`` is the maximum for ``K`` classes, or
    differential entropy for Gaussian regression models, which can be negative.

    The threshold is set by the user rather than fitted on data, so no coverage or risk guarantee is given. In
    particular, if the model is miscalibrated, accepted predictions may be wrong more often than the threshold
    suggests.

    Attributes:
        threshold: Maximum uncertainty criterion at which a prediction is accepted.
    """

    threshold: float

    def __init__(self, threshold: float) -> None:
        """Initialize the selector.

        Args:
            threshold: Maximum uncertainty criterion at which a prediction is accepted. ``inf`` accepts every
                prediction and ``-inf`` rejects every prediction.

        Raises:
            ValueError: If ``threshold`` is NaN.
        """
        threshold = float(threshold)
        if math.isnan(threshold):
            msg = "threshold must not be NaN."
            raise ValueError(msg)
        self.threshold = threshold

    @override
    def select(self, uncertainty: Any) -> Any:
        return uncertainty <= self.threshold


class SelectivePredictor[**In, R: Representation]:
    """Selective predictor for models transformed by probly.

    A selective predictor wraps an uncertainty-aware model, e.g. one transformed by probly into an ensemble or an
    MC-dropout model, and decides per instance whether to accept its prediction or to abstain. :meth:`predict` builds
    the model's representation once, quantifies its uncertainty with :func:`~probly.quantification.quantify`, derives
    the decision with :attr:`decider`, and lets :attr:`selector` determine which predictions are accepted. Since the
    criterion and the decision are computed from the same representation, they refer to the same forward passes.

    The criterion is the component of the uncertainty decomposition selected by ``notion``: the total uncertainty by
    default, or its aleatoric or epistemic part. Which decomposition is used is up to
    :func:`~probly.quantification.quantify`; for a sample of categorical distributions, e.g. the predictions of an
    ensemble, it is the entropy decomposition. Alternatively, ``notion`` can be a function that computes the
    criterion from the representation itself, e.g. one minus the maximum mean probability.

    For models not transformed by probly, compute the criterion and the decision directly and apply a
    :class:`Selector` to the criterion.

    Attributes:
        model: The wrapped uncertainty-aware model.
        selector: The rule that decides which predictions are accepted.
        representer: The representer that builds representations from the model's predictions.
        notion: The notion of uncertainty the criterion measures, or the function computing the criterion from the
            representation.
        decider: Function mapping a representation to the decision.
    """

    model: Predictor[In, Any]
    selector: Selector
    representer: Representer[Any, In, Any, R]
    notion: type[Notion] | Callable[[R], Any]
    decider: Callable[[R], Any]

    def __init__(
        self,
        model: Predictor[In, Any],
        selector: Selector,
        *,
        notion: NotionKey | Callable[[R], Any] = "total",
        decider: Callable[[R], Any] | None = None,
        representer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the selective predictor.

        Args:
            model: An uncertainty-aware model accepted by :func:`~probly.representer.representer`, e.g. an
                ensemble, an MC-dropout model, or an evidential model.
            selector: The rule that decides which predictions are accepted, e.g. a :class:`ThresholdSelector`.
            notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"``, or ``"epistemic"``.
                Defaults to ``"total"``. The model's uncertainty decomposition has to contain this notion; a single
                categorical distribution, for instance, only provides the total uncertainty. Alternatively, a
                function mapping the representation to the criterion per instance, where higher means more
                uncertain; it is used instead of :func:`~probly.quantification.quantify`.
            decider: Function mapping a representation to the decision. Defaults to
                :func:`~probly.decider.categorical_from_mean`.
            representer_kwargs: Keyword arguments passed on to :func:`~probly.representer.representer` when building
                the model's representer. Sampling-based models such as MC-dropout models require
                ``{"num_samples": ...}`` here.

        Raises:
            ValueError: If ``notion`` is a string that is not the name of a notion of uncertainty.
            TypeError: If ``notion`` is neither a string, a notion class, nor a callable.
        """
        if isinstance(notion, str):
            try:
                notion = notion_registry[cast("NotionName", notion)]
            except KeyError:
                msg = f"notion must be 'total', 'aleatoric', or 'epistemic', got {notion!r}."
                raise ValueError(msg) from None
        elif not callable(notion):
            msg = f"notion must be a string, a notion class, or a callable, got {type(notion).__name__}."
            raise TypeError(msg)

        self.model = model
        self.selector = selector
        self.representer = representer(model, **(representer_kwargs or {}))
        self.notion = notion
        self.decider = categorical_from_mean if decider is None else decider

    @final
    def predict(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Predict and decide per instance whether to abstain.

        The arguments are passed on to the model's representer.

        Returns:
            The decision, uncertainty criterion, and acceptance mask for the input.

        Raises:
            TypeError: If :attr:`notion` is a notion class and :func:`~probly.quantification.quantify` does not
                return a decomposition for the model's representation.
            KeyError: If :attr:`notion` is a notion class that the model's uncertainty decomposition does not contain.
        """
        representation = self.representer.represent(*args, **kwargs)
        uncertainty = self._criterion(representation)
        return SelectivePrediction(
            decision=self.decider(representation),
            uncertainty=uncertainty,
            accepted=self.selector.select(uncertainty),
        )

    def _criterion(self, representation: R) -> Any:  # noqa: ANN401
        notion = self.notion
        if not isinstance(notion, type):
            return notion(representation)
        decomposition = quantify(representation)
        if not isinstance(decomposition, Decomposition):
            msg = f"Expected quantify to return a Decomposition, got {type(decomposition).__name__}."
            raise TypeError(msg)
        return decomposition[notion]

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Alias for :meth:`predict`."""
        return self.predict(*args, **kwargs)
