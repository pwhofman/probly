"""Selective predictors that answer or abstain based on an uncertainty criterion.

A selective predictor is allowed to abstain: for each instance, it either returns the prediction of the wrapped model
or refrains from predicting. The choice is made by a rule on an uncertainty criterion computed from the model's
representation, which is why the wrapped model should be uncertainty-aware.

The module separates the selection rule from the pipeline that computes the criterion. A :class:`Selector` decides
on arrays of uncertainty values alone and can therefore be used with any model or with precomputed scores.
:class:`SelectivePredictor` wraps a model transformed by probly, or a plain classifier declared with
:func:`~probly.method.cast`, computes the criterion and the decision from its representation, and delegates the
selection to a :class:`Selector`.

For classifiers whose representation is categorical or a Dirichlet distribution, e.g. an ensemble, an MC-dropout
model or an evidential model, the default criterion comes from the zero-one loss: it is one minus the maximum mean
probability, the model's own probability that its decision is wrong. A :class:`ThresholdSelector` with threshold
``c`` on this criterion is Chow's rule (Chow, 1970) for an abstention cost ``c``. Other representations, e.g. credal
sets or the predictions of regression models, keep their own default decomposition, so the criterion and the meaning
of a threshold change with the model family.

The total uncertainty is the recommended criterion for selective prediction (Hofman et al., 2025); the epistemic
uncertainty suits the rejection of out-of-distribution instances, see :class:`SelectivePredictor`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, cast, final, override

from probly.decider import categorical_from_mean
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
    from probly.representation import Representation
    from probly.representer import Representer

_ZERO_ONE_LOSS = ZeroOneLoss()


class _DefaultLoss:
    """Default of the ``loss`` argument: the zero-one loss where it applies and quantify for other representations.

    Unlike a zero-one loss passed explicitly, it does not raise for representations the loss does not apply to.
    """

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "ZeroOneLoss()"


_DEFAULT_LOSS = _DefaultLoss()


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
        """Fraction of instances whose prediction is accepted, or NaN for an empty batch.

        The fraction is converted to a Python float, so the property cannot be used inside a traced function such as
        one compiled with ``jax.jit``; compute the mean of :attr:`accepted` there instead.
        """
        accepted: Any = self.accepted
        num_instances = math.prod(accepted.shape)
        if num_instances == 0:
            return math.nan
        return float(accepted.sum()) / num_instances


class Selector(ABC):
    """Base class for selection rules.

    A selector decides per instance whether a prediction is accepted, based only on its uncertainty criterion. It does
    not know the model the criterion comes from, so it works with the criterion computed by a
    :class:`SelectivePredictor` as well as with scores computed by the user for any other model. The criterion follows
    the convention of :func:`~probly.evaluation.selective_prediction.selective_prediction`: higher values mean more
    uncertain, so the instances with the largest criterion are rejected first. A confidence score has to be negated.

    The criterion is a one-dimensional array with one value per instance; inputs with a larger batch shape have to be
    flattened. A NaN criterion counts as maximally uncertain, so selectors reject it instead of raising, and a single
    failed instance does not abort the whole batch. Subclasses check the criterion with :meth:`_check_uncertainty`.

    Types of selective prediction differ in their selector: each is a subclass that overrides :meth:`select`.
    Selectors whose rule has to be fitted on data (e.g. a threshold chosen for a target coverage or risk) should
    implement the :class:`~probly.calibrator.Calibrator` protocol on arrays of the criterion and raise a
    ``ValueError`` from :meth:`select` while they are not calibrated.
    """

    @abstractmethod
    def select(self, uncertainty: Any) -> Any:  # noqa: ANN401
        """Decide which predictions are accepted.

        Args:
            uncertainty: The uncertainty criterion per instance, as a one-dimensional array.

        Returns:
            A boolean array with the shape of ``uncertainty``; ``True`` means the prediction is accepted.

        Raises:
            TypeError: If ``uncertainty`` is not an array.
            ValueError: If ``uncertainty`` is not one-dimensional.
        """
        raise NotImplementedError

    @staticmethod
    def _check_uncertainty[U](uncertainty: U) -> U:
        """Check that the criterion is a one-dimensional array and return it unchanged."""
        ndim = getattr(uncertainty, "ndim", None)
        if ndim is None:
            msg = f"uncertainty must be an array with one value per instance, got {type(uncertainty).__name__}."
            raise TypeError(msg)
        if ndim != 1:
            shape = tuple(cast("Any", uncertainty).shape)
            msg = f"uncertainty must be one-dimensional with one value per instance, got shape {shape}."
            raise ValueError(msg)
        return uncertainty

    def __call__(self, uncertainty: Any) -> Any:  # noqa: ANN401
        """Alias for :meth:`select`."""
        return self.select(uncertainty)


class ThresholdSelector(Selector):
    """Selector that accepts a prediction if its uncertainty does not exceed a fixed threshold.

    A prediction is accepted if and only if its uncertainty criterion is less than or equal to the threshold, so
    ties at the threshold are accepted and NaN criteria are rejected. The threshold lives on the scale of the
    criterion, e.g. one minus the maximum probability for the default criterion of :class:`SelectivePredictor` in
    classification, which lies in ``[0, 1 - 1/K]`` for ``K`` classes, entropy in nats under the log loss, or
    differential entropy for Gaussian regression models, which can be negative.

    On the default criterion for categorical representations, the threshold is the cost of abstaining: if an
    accepted prediction costs 1 when it is wrong and 0 when it is right, and an abstention costs ``c``, then
    ``ThresholdSelector(c)`` is Chow's rule (Chow, 1970). It abstains exactly where the model's own probability of an
    error exceeds ``c`` and minimizes the expected cost if the model's probabilities are calibrated. Since the
    criterion never exceeds ``1 - 1/K``, costs of at least ``1 - 1/K`` never lead to an abstention. Under another
    proper loss, such as the log loss, the criterion is the expected loss of the model's best prediction and the
    threshold is the abstention cost on the scale of that loss. On other representations, e.g. the upper entropy of
    a credal set, the threshold has no such reading.

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
        return self._check_uncertainty(uncertainty) <= self.threshold


class SelectivePredictor[**In, R: Representation]:
    """Selective predictor for models transformed by probly.

    A selective predictor wraps an uncertainty-aware model, e.g. one transformed by probly into an ensemble or an
    MC-dropout model, and decides per instance whether to accept its prediction or to abstain. :meth:`predict` builds
    the model's representation once, decomposes its uncertainty, derives the decision with :attr:`decider`, and lets
    :attr:`selector` determine which predictions are accepted. Since the criterion and the decision are computed from
    the same representation, they refer to the same forward passes.

    The criterion is the component of the uncertainty decomposition selected by ``notion``: the total uncertainty by
    default, or its aleatoric or epistemic part. For categorical predictions, the decomposition is induced by the
    task loss :attr:`loss`, the loss by which the predictions are evaluated, see
    :class:`~probly.quantification.decomposition.scoring_rule.SecondOrderScoringRuleDecomposition`. The total
    uncertainty is then the expected task loss of the model's best prediction under its mean prediction, so the
    criterion is aligned with the evaluation. The default zero-one loss gives, with ``theta_bar`` the mean of the
    predicted distributions:

    - total uncertainty ``1 - max_k theta_bar_k``, the model's own probability that the decision of the default
      decider is wrong, so that a :class:`ThresholdSelector` on it is Chow's rule;
    - aleatoric uncertainty ``E[1 - max_k theta_k]``, the expected error probability of the individual predictions;
    - epistemic uncertainty, their difference, the expected disagreement with the decision.

    The log loss gives the entropy decomposition (entropy of the mean, expected entropy, and mutual information), and
    the Brier loss the Gini decomposition. Which decomposition is used depends on the representation:

    - samples of categorical distributions, e.g. the predictions of an ensemble or an MC-dropout model: the
      decomposition induced by :attr:`loss`;
    - a single categorical distribution, e.g. of a plain classifier: the expected loss of the best prediction, which
      is a total uncertainty only;
    - Dirichlet distributions, e.g. of evidential models: the zero-one decomposition; other losses raise a
      ``NotImplementedError``;
    - any other representation, e.g. of a regression model or a credal set: the decomposition of
      :func:`~probly.quantification.quantify` if ``loss`` is left at its default; a loss passed explicitly raises a
      ``NotImplementedError`` there instead of being ignored.

    Note that the loss takes precedence over a decomposition that a method registers for its own representation,
    whenever that representation is also one of the first three. The decompositions of DARE, SNGP and
    heteroscedastic networks, and the vacuity of evidential models, are therefore used only if :attr:`loss` is
    ``None``, in which case every representation is decomposed by :func:`~probly.quantification.quantify`.
    Alternatively, ``notion`` can be a function that computes the criterion from the representation itself.

    For selective prediction, the total uncertainty is the recommended notion (Hofman et al., 2025). The epistemic
    uncertainty ignores the noise in the labels, so it rejects errors less reliably, but it suits the rejection of
    out-of-distribution instances, for which the log loss, i.e. mutual information, separates better than the
    zero-one loss.

    A plain classifier can be wrapped once its output is declared with :func:`~probly.method.cast`, e.g.
    ``cast(net, predictor_type="logit_classifier")`` for a network that outputs logits, or
    ``cast(forest, predictor_type="probabilistic_classifier")`` for a scikit-learn classifier. Its representation is a
    single categorical distribution, so the default criterion is one minus its maximum probability. For precomputed
    scores and for models that :func:`~probly.method.cast` does not cover, compute the criterion directly and apply a
    :class:`Selector` to it.

    Attributes:
        model: The wrapped uncertainty-aware model.
        selector: The rule that decides which predictions are accepted.
        representer: The representer that builds representations from the model's predictions.
        notion: The notion of uncertainty the criterion measures, or the function computing the criterion from the
            representation.
        loss: The task loss, which induces the uncertainty decomposition of categorical predictions, or ``None`` to
            use :func:`~probly.quantification.quantify` for every representation.
        decider: Function mapping a representation to the decision.
    """

    model: Predictor[In, Any]
    selector: Selector
    representer: Representer[Any, In, Any, R]
    notion: type[Notion] | Callable[[R], Any]
    loss: ScoringRule | None
    decider: Callable[[R], Any]
    _loss_is_default: bool

    def __init__(
        self,
        model: Predictor[In, Any],
        selector: Selector,
        *,
        notion: NotionKey | Callable[[R], Any] = "total",
        loss: ScoringRule | _DefaultLoss | None = _DEFAULT_LOSS,
        decider: Callable[[R], Any] | None = None,
        representer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the selective predictor.

        Args:
            model: A model accepted by :func:`~probly.representer.representer`, e.g. an ensemble, an MC-dropout
                model, an evidential model, or a plain classifier passed through :func:`~probly.method.cast`.
            selector: The rule that decides which predictions are accepted, e.g. a :class:`ThresholdSelector`.
            notion: The notion of uncertainty to select on, e.g. ``"total"``, ``"aleatoric"``, or ``"epistemic"``.
                Defaults to ``"total"``. The model's uncertainty decomposition has to contain this notion; a single
                categorical distribution, for instance, only provides the total uncertainty, and the decomposition of
                DDU provides no total uncertainty, so it needs ``"aleatoric"`` or ``"epistemic"``. Alternatively, a
                function mapping the representation to the criterion per instance, where higher means more
                uncertain; it is used instead of the decomposition.
            loss: The task loss, i.e. the loss by which the predictions are evaluated. It induces the decomposition
                of categorical predictions, e.g. :class:`~probly.quantification.scoring_rule.ZeroOneLoss` (the
                default), :class:`~probly.quantification.scoring_rule.LogLoss` for the entropy decomposition, or
                :class:`~probly.quantification.scoring_rule.BrierLoss`. If omitted, representations the zero-one loss
                does not apply to, e.g. of regression models or credal sets, are decomposed by
                :func:`~probly.quantification.quantify`. A loss passed explicitly has to apply to the representation,
                otherwise :meth:`predict` raises. ``None`` uses :func:`~probly.quantification.quantify` for every
                representation, including method-specific decompositions. Ignored if ``notion`` is a function.
            decider: Function mapping a representation to the decision. Defaults to
                :func:`~probly.decider.categorical_from_mean`.
            representer_kwargs: Keyword arguments passed on to :func:`~probly.representer.representer` when building
                the model's representer. Sampling-based models such as MC-dropout models require
                ``{"num_samples": ...}`` here.

        Raises:
            ValueError: If ``notion`` is a string that is not the name of a notion of uncertainty.
            TypeError: If ``notion`` is neither a string, a notion class, nor a callable, e.g. a class that is not a
                subclass of :class:`~probly.quantification.notion.Notion`, or if ``loss`` is neither a scoring rule nor
                ``None``.
        """
        if isinstance(notion, str):
            try:
                notion = notion_registry[cast("NotionName", notion)]
            except KeyError:
                names = ", ".join(repr(name) for name in notion_registry)
                msg = f"notion must be one of {names}, got {notion!r}."
                raise ValueError(msg) from None
        elif isinstance(notion, type):
            if not issubclass(notion, Notion):
                msg = f"notion must be a subclass of Notion if it is a class, got {notion.__name__}."
                raise TypeError(msg)
        elif not callable(notion):
            msg = f"notion must be a string, a notion class, or a callable, got {type(notion).__name__}."
            raise TypeError(msg)
        loss_is_default = loss is _DEFAULT_LOSS
        if loss_is_default:
            loss = _ZERO_ONE_LOSS
        if loss is not None and not isinstance(loss, ScoringRule):
            msg = f"loss must be a scoring rule or None, got {type(loss).__name__}."
            raise TypeError(msg)

        self.model = model
        self.selector = selector
        self.representer = representer(model, **(representer_kwargs or {}))
        self.notion = notion
        self.loss = loss
        self._loss_is_default = loss_is_default
        self.decider = categorical_from_mean if decider is None else decider

    @final
    def predict(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Predict and decide per instance whether to abstain.

        The arguments are passed on to the model's representer. Gradients are not stopped: with PyTorch, the
        criterion and the decision can be part of the autograd graph, so that ``.numpy()`` fails on them. Wrap
        inference in ``torch.no_grad()``.

        Returns:
            The decision, uncertainty criterion, and acceptance mask for the input.

        Raises:
            TypeError: If :attr:`notion` is a notion class and :func:`~probly.quantification.quantify` does not
                return a decomposition for the model's representation.
            KeyError: If :attr:`notion` is a notion class that the model's uncertainty decomposition does not contain.
            NotImplementedError: If :attr:`loss` was passed explicitly and does not apply to the representation: a
                loss other than the zero-one loss for a Dirichlet distribution, or any loss for a representation that
                is neither categorical nor a Dirichlet distribution.
            ValueError: If the criterion is not one-dimensional and :attr:`selector` checks its shape, as
                :class:`ThresholdSelector` does.
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
        decomposition = self._decomposition(representation)
        if not isinstance(decomposition, Decomposition):
            msg = f"Expected quantify to return a Decomposition, got {type(decomposition).__name__}."
            raise TypeError(msg)
        return decomposition[notion]

    def _decomposition(self, representation: R) -> Any:  # noqa: ANN401
        loss = self.loss
        if loss is None:
            return quantify(representation)
        if isinstance(representation, CategoricalDistributionSample):
            if isinstance(loss, ZeroOneLoss):
                # Computes the epistemic part directly instead of as total minus aleatoric, which leaves float noise
                # around zero for instances where all members agree.
                return SecondOrderZeroOneDecomposition(representation)
            return SecondOrderScoringRuleDecomposition(representation, loss)
        if isinstance(representation, CategoricalDistribution):
            # A single distribution is evaluated as a sample with one member, which reuses the backend implementations.
            return ConstantTotalDecomposition(generalized_entropy_of_expected(create_sample([representation]), loss))
        if isinstance(representation, DirichletDistribution):
            if not isinstance(loss, ZeroOneLoss):
                msg = (
                    f"{type(loss).__name__} is not supported for Dirichlet distributions; "
                    "use ZeroOneLoss() or loss=None."
                )
                raise NotImplementedError(msg)
            return SecondOrderZeroOneDecomposition(representation)
        if self._loss_is_default:
            return quantify(representation)
        msg = (
            f"{type(loss).__name__} is not supported for {type(representation).__name__}; "
            "omit loss to use the default decomposition of this representation, or pass loss=None."
        )
        raise NotImplementedError(msg)

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> SelectivePrediction[Any]:
        """Alias for :meth:`predict`."""
        return self.predict(*args, **kwargs)
