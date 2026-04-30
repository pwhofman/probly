"""laplace-torch-specific predictor dispatch."""

from __future__ import annotations

from laplace.baselaplace import BaseLaplace

from probly.representation.distribution import CategoricalDistribution, create_categorical_distribution

from ._common import predict, predict_raw


@predict.register(BaseLaplace)
def _[**In](predictor: BaseLaplace, *args: In.args, **kwargs: In.kwargs) -> CategoricalDistribution:
    """Wrap a laplace-torch classification output as a :class:`CategoricalDistribution`.

    Forwards args/kwargs to ``BaseLaplace.__call__`` (see laplace-torch docs). Regression raises.
    """
    if predictor.likelihood != "classification":
        msg = f"only likelihood='classification' is supported, got {predictor.likelihood!r}"
        raise NotImplementedError(msg)
    return create_categorical_distribution(predict_raw(predictor, *args, **kwargs))  # ty:ignore[invalid-return-type]
