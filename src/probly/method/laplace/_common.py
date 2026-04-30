"""Shared Laplace approximation interface and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch
from probly.method.method import predictor_transformation
from probly.predictor import Predictor, RandomPredictor, predict, predict_raw
from probly.representation.distribution import (
    CategoricalDistribution,
    create_categorical_distribution,
)

if TYPE_CHECKING:
    from laplace.baselaplace import BaseLaplace


@runtime_checkable
class LaplacePredictor[**In, Out](Predictor[In, Out], Protocol):
    """Common Laplace interface.

    Exposes the underlying laplace-torch object and a fit method. Concrete
    Laplace predictors implement either GLM closed-form or Monte Carlo
    posterior-sampling prediction; see :class:`LaplaceGLMPredictor` and
    :class:`LaplaceMCPredictor`.
    """

    la: BaseLaplace

    def fit(
        self,
        loader: object,
        optimize_prior: bool = False,
        **kwargs: object,
    ) -> None:
        """Fit the Laplace posterior on a data loader.

        Args:
            loader: A torch data loader yielding ``(x, y)`` batches.
            optimize_prior: If ``True``, also call
                ``la.optimize_prior_precision(**kwargs)`` after fitting.
            **kwargs: Forwarded to ``optimize_prior_precision`` when
                ``optimize_prior=True``.
        """
        ...


@runtime_checkable
class LaplaceGLMPredictor[**In, Out](LaplacePredictor[In, Out], Protocol):
    """Closed-form Laplace via GLM linearization.

    Predictions are computed in closed form (Gaussian for regression,
    probit-approximated probabilities for classification). Does not
    extend :class:`RandomPredictor`; calling ``.sample(...)`` on a GLM
    predictor raises ``AttributeError``.
    """


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class LaplaceMCPredictor[**In, Out](LaplacePredictor[In, Out], RandomPredictor[In, Out], Protocol):
    """Monte Carlo posterior-sampling Laplace.

    Predictions are computed by sampling weights from the Laplace
    posterior and forwarding through the network. ``.predict(...)``
    returns the MC mean; ``.sample(...)`` returns the raw posterior
    samples.
    """


@flexdispatch
def laplace_generator[**In, Out](
    base: Predictor[In, Out],
    pred_type: str = "glm",
    **kwargs: object,
) -> LaplacePredictor[In, Out]:
    """Generate a Laplace predictor from a base model. Dispatch entry point."""
    msg = f"No Laplace generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=False)
@LaplacePredictor.register_factory
def laplace[**In, Out](
    base: Predictor[In, Out],
    pred_type: str = "glm",
    **kwargs: object,
) -> LaplacePredictor[In, Out]:
    """Wrap a trained model in a Laplace approximation.

    A thin pass-through over the
    `laplace-torch <https://github.com/aleximmer/Laplace>`_ package
    (Daxberger et al., 2021, "Laplace Redux"). All keyword arguments
    flow directly to ``laplace.Laplace(model, **kwargs)`` -- consult the
    laplace-torch documentation for the full configuration surface
    (``subset_of_weights``, ``hessian_structure``, ``prior_precision``,
    ``sigma_noise``, ``temperature``, ...).

    The returned predictor is unfitted. Call ``.fit(loader)`` to fit the
    posterior on data before predicting.

    Args:
        base: The trained model to wrap.
        pred_type: Either ``"glm"`` (closed-form GLM linearization,
            returns :class:`LaplaceGLMPredictor`) or ``"nn"`` (Monte
            Carlo posterior sampling, returns :class:`LaplaceMCPredictor`).
        **kwargs: Forwarded to ``laplace.Laplace(model, **kwargs)``.

    Returns:
        An unfitted Laplace predictor.

    Raises:
        ValueError: If ``pred_type`` is not ``"glm"`` or ``"nn"``.

    Example:
        >>> la_pred = laplace(model, pred_type="glm",
        ...                   likelihood="classification",
        ...                   subset_of_weights="last_layer",
        ...                   hessian_structure="kron")
        >>> la_pred.fit(train_loader)
        >>> probs = la_pred.predict(x, link_approx="probit")
    """
    if pred_type not in {"glm", "nn"}:
        msg = f"pred_type must be 'glm' or 'nn', got {pred_type!r}"
        raise ValueError(msg)
    return laplace_generator(base, pred_type=pred_type, **kwargs)


@predict.register(LaplaceMCPredictor)
def _predict_mc[**In](
    predictor: LaplaceMCPredictor[In, CategoricalDistribution],
    *args: In.args,
    **kwargs: In.kwargs,
) -> CategoricalDistribution:
    """Wrap the MC mean output of a Laplace predictor in a CategoricalDistribution.

    Classification only — for regression this raises
    :class:`NotImplementedError`. Use the predictor's ``.sample(x)`` method or
    ``probly.representer.representer(predictor, num_samples=N)`` to obtain
    raw posterior samples instead.

    GLM has its own auto-dispatch path: ``LaplaceGLMPredictor`` provides
    ``predict_representation`` and is therefore handled by the generic
    ``RepresentationPredictor`` ``predict`` registration.
    """
    likelihood = getattr(predictor.la, "likelihood", None)
    if likelihood != "classification":
        msg = (
            "predict() distribution wrapping is only implemented for "
            f"likelihood='classification', got {likelihood!r}. "
            "Call predictor.predict(x) directly for raw output."
        )
        raise NotImplementedError(msg)
    return create_categorical_distribution(predict_raw(predictor, *args, **kwargs))
