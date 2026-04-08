"""Shared conformalized credal set predictor implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.predictor import Predictor


@lazydispatch
def conformalized_credal_set_factory[T: Predictor](base: T) -> T:
    """Create a conformalized credal set predictor from a base model."""
    msg = f"No conformalized credal set predictor registered for type {type(base)}"
    raise NotImplementedError(msg)


def register(cls: LazyType, factory: Callable) -> None:
    """Register a class which can be used as a base for conformalized credal set prediction.

    Args:
        cls: The lazy type to register.
        factory: The factory function that wraps the model.

    """
    conformalized_credal_set_factory.register(cls=cls, func=factory)


def conformalized_credal_set[T: Predictor](base: T) -> T:
    """Create a conformalized credal set predictor from a base predictor.

    The returned predictor must be calibrated via its ``calibrate`` method before
    calling ``predict``.  After calibration, ``predict`` returns a
    :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`
    for every batch of test inputs.

    The calibration procedure uses LAC (Least Ambiguous set-valued Classifier)
    nonconformity scores: for each calibration sample :math:`(x_i, y_i)`, the
    score is :math:`s_i = 1 - \\hat{p}(y_i | x_i)`.  Given significance level
    :math:`\\alpha`, the conformal threshold is the
    :math:`\\lceil (n+1)(1-\\alpha) \\rceil / n`-quantile of those scores.

    At prediction time, for a test point :math:`x` with softmax probabilities
    :math:`\\hat{p} = (\\hat{p}_1, \\dots, \\hat{p}_K)`, the credal set is the
    probability-interval set

    .. math::

        C(x) = \\{p \\in \\Delta^{K-1} :
                 \\max(0,\\, \\hat{p}_k - \\hat{q}) \\le p_k \\le
                 \\min(1,\\, \\hat{p}_k + \\hat{q}),\\; k=1,\\dots,K\\}

    where :math:`\\hat{q}` is the calibrated threshold.

    Args:
        base: The base model to be used for the conformalized credal set predictor.

    Returns:
        A predictor whose ``predict`` method returns
        :class:`~probly.representation.credal_set.torch.TorchProbabilityIntervalsCredalSet`.

    """
    return conformalized_credal_set_factory(base)
