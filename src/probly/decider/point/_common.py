"""Common deciders for point predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch
import numpy as np

from probly.representation.distribution._common import GaussianDistribution
from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.representation import Representation


@flexdispatch
def point_from_mean(representation: Representation | ArrayLike | np.ndarray) -> ArrayLike | np.ndarray:
    """Reduce a representation to a point prediction through its mean.

    This is the Bayes decision under squared error loss. Arrays and tensors are already point
    predictions and pass through unchanged, a sample of point predictions is reduced to its sample
    mean, and a Gaussian distribution is reduced to its mean.
    """
    msg = f"point_from_mean decider not supported for {type(representation).__name__}."
    raise NotImplementedError(msg)


@point_from_mean.register(np.ndarray)
def _(prediction: np.ndarray) -> np.ndarray:
    # Registered on the concrete array type only: probly's representations implement the
    # array-like protocol themselves and must reach their own registrations below.
    return prediction


@point_from_mean.register(Sample)
def _(sample: Sample) -> ArrayLike | np.ndarray:
    return sample.sample_mean()


@point_from_mean.register(GaussianDistribution)
def _(gaussian: GaussianDistribution) -> ArrayLike | np.ndarray:
    return gaussian.mean
