"""Selective predictors that answer or abstain based on an uncertainty criterion.

A selective predictor is allowed to abstain: for each instance, it either returns the prediction of the wrapped model
or refrains from predicting. The choice is made by a rule on an uncertainty criterion computed from the model's
representation, which is why the wrapped model has to be uncertainty-aware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, final, override

from probly.decider import categorical_from_mean
from probly.quantification import Decomposition, notion_registry, quantify
from probly.representer import representer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from probly.predictor import Predictor
    from probly.quantification import Notion
    from probly.quantification.notion import NotionKey
    from probly.representation import Representation
    from probly.representer import Representer


@dataclass(frozen=True, slots=True)
class SelectivePrediction[D]:
    """Output of a selective predictor.

    The decision is returned for every instance, including those the predictor abstains on, so that rejected
    predictions remain available for inspection and evaluation.

    Attributes:
        decision: The decision derived from the model's representation, e.g. a categorical distribution whose argmax
            is the predicted class.
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


class SelectivePredictor[**In, R: Representation](ABC):
    """Base class for selective predictors.

    A selective predictor wraps an uncertainty-aware model, e.g. one transformed by probly into an ensemble or an
    MC-dropout model, and decides per instance whether to accept its prediction or to abstain. :meth:`predict` builds
    the model's representation once, quantifies its uncertainty with :func:`~probly.quantification.quantify`, derives
    the decision with :attr:`decider`, and lets :meth:`select` determine which predictions are accepted. Since the
    criterion and the decision are computed from the same representation, they refer to the same forward passes.

    The criterion is the component of the uncertainty decomposition selected by ``notion``: the total uncertainty by
    default, or its aleatoric or epistemic part. Which decomposition is used is up to
    :func:`~probly.quantification.quantify`; for a sample of categorical distributions, e.g. the predictions of an
    ensemble, it is the entropy decomposition. The criterion follows the convention of
    :func:`~probly.evaluation.selective_prediction.selective_prediction`: higher values mean more uncertain, so the
    instances with the largest criterion are rejected first.

    Types of selective predictors differ only in how they select: each is a subclass that overrides :meth:`select`,
    while :meth:`predict` stays fixed. Subclasses whose selection has to be fitted on data (e.g. a threshold chosen
    for a target coverage or risk) should implement the :class:`~probly.calibrator.Calibrator` protocol and raise a
    ``ValueError`` from :meth:`select` while they are not calibrated.

    Attributes:
        model: The wrapped uncertainty-aware model.
        representer: The representer that builds representations from the model's predictions.
        notion: The notion of uncertainty the criterion measures.
        decider: Function mapping a representation to the decision.
    """

    model: Predictor[In, Any]
    representer: Representer[Any, In, Any, R]
    notion: type[Notion]
    decider: Callable[[R], Any]

    def __init__(
        self,
        model: Predictor[In, Any],
        *,
        notion: NotionKey = "total",
        decider: Callable[[R], Any] | None = None,
        representer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the selective predictor.

        Args:
            model: An uncertainty-aware model accepted by :func:`~probly.representer.representer`, e.g. an
                ensemble, an MC-dropout model, or an evidential model.
            notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"``, or ``"epistemic"``.
                Defaults to ``"total"``. The model's uncertainty decomposition has to contain this notion; a single
                categorical distribution, for instance, only provides the total uncertainty.
            decider: Function mapping a representation to the decision. Defaults to
                :func:`~probly.decider.categorical_from_mean`.
            representer_kwargs: Keyword arguments passed on to :func:`~probly.representer.representer` when building
                the model's representer. Sampling-based models such as MC-dropout models require
                ``{"num_samples": ...}`` here.

        Raises:
            ValueError: If ``notion`` is not the name of a notion of uncertainty.
        """
        if isinstance(notion, str):
            try:
                notion = notion_registry[notion]
            except KeyError:
                msg = f"notion must be 'total', 'aleatoric', or 'epistemic', got {notion!r}."
                raise ValueError(msg) from None

        self.model = model
        self.representer = representer(model, **(representer_kwargs or {}))
        self.notion = notion
        self.decider = categorical_from_mean if decider is None else decider

    @abstractmethod
    def select(self, uncertainty: Any) -> Any:  # noqa: ANN401
        """Decide which predictions are accepted.

        Args:
            uncertainty: The uncertainty criterion per instance.

        Returns:
            A boolean array with the batch shape of ``uncertainty``; ``True`` means the prediction is accepted.
        """
        raise NotImplementedError

    @final
    def predict(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Predict and decide per instance whether to abstain.

        The arguments are passed on to the model's representer.

        Returns:
            The decision, uncertainty criterion, and acceptance mask for the input.

        Raises:
            TypeError: If :func:`~probly.quantification.quantify` does not return a decomposition for the model's
                representation.
            KeyError: If the model's uncertainty decomposition does not contain :attr:`notion`.
        """
        representation = self.representer.represent(*args, **kwargs)
        decomposition = quantify(representation)
        if not isinstance(decomposition, Decomposition):
            msg = f"Expected quantify to return a Decomposition, got {type(decomposition).__name__}."
            raise TypeError(msg)
        uncertainty = decomposition[self.notion]
        return SelectivePrediction(
            decision=self.decider(representation),
            uncertainty=uncertainty,
            accepted=self.select(uncertainty),
        )

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Alias for :meth:`predict`."""
        return self.predict(*args, **kwargs)


class ThresholdSelectivePredictor[**In, R: Representation](SelectivePredictor[In, R]):
    """Selective predictor that accepts a prediction if its uncertainty does not exceed a fixed threshold.

    A prediction is accepted if and only if its uncertainty criterion is less than or equal to the threshold, so
    ties at the threshold are accepted. The threshold lives on the scale of the criterion, e.g. entropy in nats for
    classification, where ``log(K)`` is the maximum for ``K`` classes, or differential entropy for Gaussian
    regression models, which can be negative.

    The threshold is set by the user rather than fitted on data, so no coverage or risk guarantee is given. In
    particular, if the model is miscalibrated, accepted predictions may be wrong more often than the threshold
    suggests.

    Attributes:
        threshold: Maximum uncertainty criterion at which a prediction is accepted.
    """

    threshold: float

    def __init__(
        self,
        model: Predictor[In, Any],
        threshold: float,
        *,
        notion: NotionKey = "total",
        decider: Callable[[R], Any] | None = None,
        representer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the selective predictor.

        Args:
            model: An uncertainty-aware model accepted by :func:`~probly.representer.representer`.
            threshold: Maximum uncertainty criterion at which a prediction is accepted. ``inf`` accepts every
                prediction and ``-inf`` rejects every prediction.
            notion: The notion of uncertainty to select on; see :class:`SelectivePredictor`.
            decider: Function mapping a representation to the decision; see :class:`SelectivePredictor`.
            representer_kwargs: Keyword arguments for the model's representer; see :class:`SelectivePredictor`.

        Raises:
            ValueError: If ``threshold`` is NaN or ``notion`` is not the name of a notion of uncertainty.
        """
        threshold = float(threshold)
        if math.isnan(threshold):
            msg = "threshold must not be NaN."
            raise ValueError(msg)
        super().__init__(model, notion=notion, decider=decider, representer_kwargs=representer_kwargs)
        self.threshold = threshold

    @override
    def select(self, uncertainty: Any) -> Any:
        return uncertainty <= self.threshold
