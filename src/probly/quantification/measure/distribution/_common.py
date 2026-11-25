"""Common definitions of distribution measures."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

from flextype import flexdispatch

from probly.quantification._quantification import measure
from probly.representation.distribution import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
    Distribution,
    GaussianDistributionSample,
)
from probly.representation.sample import Sample

DEFAULT_MEAN_FIELD_FACTOR: float = math.pi / 8.0

if TYPE_CHECKING:
    from ndlike.array_like import WeakArrayLike

    from probly.lazy_types import NativeArray
    from probly.metrics import ScoringRule
    from probly.representation.distribution import SecondOrderDistribution

type LogBase = float | Literal["normalize"] | None

DEFAULT_NUM_SAMPLES: int = 1000

# Bisection steps for the simplex-constrained quantile solver behind ``min_expected_total_variation``
# (the level is bracketed in [0, 1], so n steps give ~2**-n precision). Shared across backends.
TOTAL_VARIATION_BISECTION_ITERATIONS: int = 60


@measure.register(CategoricalDistribution)
@flexdispatch
def entropy(distribution: Distribution, base: LogBase = None) -> WeakArrayLike:
    """Compute the entropy of a distribution."""
    msg = f"Entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> WeakArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def mutual_information(distribution: SecondOrderDistributionLike, base: LogBase = None) -> WeakArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def generalized_entropy_of_expected(
    distribution: SecondOrderDistributionLike, scoring_rule: ScoringRule
) -> WeakArrayLike:
    """Compute the generalized entropy ``G(theta_bar)`` of the BMA under a per-instance scoring rule."""
    msg = f"Generalized entropy of expected is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def expected_generalized_entropy(distribution: SecondOrderDistributionLike, scoring_rule: ScoringRule) -> WeakArrayLike:
    """Compute the expected generalized entropy ``E[G(theta)]`` under a per-instance scoring rule."""
    msg = f"Expected generalized entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


def _generalized_entropy(scoring_rule: ScoringRule, probabilities: NativeArray) -> NativeArray:
    """Compute ``G(theta) = scoring_rule(theta, theta)``, checking that the rule returns one loss per instance."""
    losses = scoring_rule(probabilities, probabilities)
    expected_shape = tuple(probabilities.shape[:-1])
    shape = tuple(getattr(losses, "shape", ()))
    if shape != expected_shape:
        msg = (
            f"The scoring rule must return one loss per instance, of shape {expected_shape}, got shape {shape}. "
            "Pass a per-instance rule such as brier_loss_per_instance rather than an averaged metric."
        )
        raise ValueError(msg)
    return cast("NativeArray", losses)


@flexdispatch
def max_probability_complement_of_expected(distribution: SecondOrderDistributionLike) -> WeakArrayLike:
    """Compute one minus the max probability of the expected value of a second-order distribution."""
    msg = f"Max probability complement of expected is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def expected_max_probability_complement(distribution: SecondOrderDistributionLike) -> WeakArrayLike:
    """Compute the expected value of one minus the max probability under a second-order distribution.

    Computed directly for a sample. For a parametric distribution such as a Dirichlet it is
    estimated by Monte-Carlo, and that implementation also takes ``num_samples`` and a ``generator``.
    """
    msg = f"Expected max probability complement is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def max_disagreement(distribution: SecondOrderDistributionLike) -> WeakArrayLike:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    msg = f"Max disagreement is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def min_expected_total_variation(distribution: SecondOrderDistributionLike) -> WeakArrayLike:
    r"""Compute the distance-based epistemic uncertainty of a second-order distribution.

    Defined by :cite:`saleSecondOrder2024` as the smallest expected total-variation distance from
    ``Q`` to a point on the simplex:

    .. math::

        EU(Q) = \tfrac{1}{2} \min_{q \in \Delta} \mathbb{E}_{p \sim Q}[\lVert p - q \rVert_1],

    bounded by ``(K - 1) / K`` for ``K`` classes. Computed directly for a sample. For a parametric
    distribution such as a Dirichlet it is estimated by Monte-Carlo, and that implementation also
    takes ``num_samples`` and a ``generator``.

    Args:
        distribution: A sample of categorical distributions or a Dirichlet distribution.
    """
    msg = f"Min expected total variation is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def vacuity(distribution: SecondOrderDistributionLike) -> WeakArrayLike:
    """Compute the vacuity of a second-order distribution.

    For a Dirichlet distribution Dir(alpha) with K classes and alpha_0 = sum_c alpha_c,
    the vacuity is defined as K / alpha_0 and lies in (0, 1]. It corresponds to the
    "I do not know" mass in Dempster-Shafer / subjective-logic theory introduced by
    :cite:`sensoyEvidentialDeep2018`, and is equivalent (up to the constant K)
    to the inverse of the predicted evidence in Natural Posterior Networks
    :cite:`charpentierNaturalPosteriorNetwork2022`.
    """
    msg = f"Vacuity is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def dempster_shafer_uncertainty(
    distribution: Distribution, mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR
) -> WeakArrayLike:
    r"""Compute the Dempster-Shafer uncertainty score of a distribution over logits.

    For a Gaussian distribution N(h, sigma^2) over K-class logits, this is the
    Dempster-Shafer / vacuity metric used by SNGP
    :cite:`liuSimplePrincipled2020` (Eq. 15) and originally introduced
    by :cite:`sensoyEvidentialDeep2018`:

    .. math::

        h^{adj}_k = \frac{h_k}{\sqrt{1 + \lambda \sigma_k^2}}, \quad
        u(x) = \frac{K}{K + \sum_k \exp(h^{adj}_k)}

    where ``lambda = mean_field_factor`` (default ``pi / 8``) controls the
    mean-field correction that shrinks the logits toward zero when the variance
    is large (e.g. for OOD inputs), driving the score toward 1.

    The score lies in (0, 1] and corresponds to the Sensoy vacuity ``K / S``
    applied to a soft-evidential Dirichlet with ``alpha = 1 + exp(h_adj)``.
    Setting ``mean_field_factor=0`` reduces it to the variance-free formula
    ``K / (K + sum_k exp(h_k))``.

    Args:
        distribution: A distribution over K-class logits whose backend has a
            registered implementation (e.g. an SNGP-style Gaussian).
        mean_field_factor: Coefficient in front of the variance in the
            mean-field denominator. Defaults to ``pi / 8`` (paper / SNGP recipe).

    Returns:
        The Dempster-Shafer uncertainty score, with the trailing class axis
        removed.
    """
    msg = f"Dempster-Shafer uncertainty is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


# The measures below combine other measures and array operations that NumPy, PyTorch and JAX share, so one
# implementation on the backend-independent representation classes serves every backend.


@entropy_of_expected_predictive_distribution.register(DirichletDistribution)
def _dirichlet_entropy_of_expected_predictive_distribution(
    distribution: DirichletDistribution, base: LogBase = None
) -> NativeArray:
    """Compute the entropy of the mean categorical distribution of a Dirichlet distribution."""
    return entropy(distribution.mean, base=base)


@entropy_of_expected_predictive_distribution.register(CategoricalDistributionSample)
def _categorical_sample_entropy_of_expected_predictive_distribution(
    sample: CategoricalDistributionSample, base: LogBase = None
) -> NativeArray:
    """Compute the entropy of the mean categorical distribution of a sample."""
    return entropy(sample.sample_mean(), base=base)


@conditional_entropy.register((CategoricalDistributionSample, GaussianDistributionSample))
def _sample_conditional_entropy(
    sample: CategoricalDistributionSample | GaussianDistributionSample, base: LogBase = None
) -> NativeArray:
    """Compute the mean entropy of the member distributions of a sample."""
    member_entropies = entropy(sample.samples, base=base)  # (M, ...)
    return cast("NativeArray", member_entropies).mean(0)


@mutual_information.register((DirichletDistribution, GaussianDistributionSample))
def _mutual_information_as_entropy_difference(
    distribution: DirichletDistribution | GaussianDistributionSample, base: LogBase = None
) -> NativeArray:
    """Compute the entropy of the expected predictive distribution minus the conditional entropy."""
    total = entropy_of_expected_predictive_distribution(distribution, base=base)
    conditional = conditional_entropy(distribution, base=base)
    return cast("NativeArray", total) - cast("NativeArray", conditional)  # ty: ignore[unsupported-operator]


@generalized_entropy_of_expected.register(CategoricalDistributionSample)
def _categorical_sample_generalized_entropy_of_expected[A: NativeArray](
    sample: CategoricalDistributionSample[CategoricalDistribution[A]], scoring_rule: ScoringRule
) -> NativeArray:
    """Compute ``G(theta_bar)`` for the mean ``theta_bar`` of a categorical sample."""
    return _generalized_entropy(scoring_rule, sample.sample_mean().probabilities)


@expected_generalized_entropy.register(CategoricalDistributionSample)
def _categorical_sample_expected_generalized_entropy[A: NativeArray](
    sample: CategoricalDistributionSample[CategoricalDistribution[A]], scoring_rule: ScoringRule
) -> NativeArray:
    """Compute ``E[G(theta)]``, the mean of ``G`` over the members ``theta`` of a categorical sample."""
    member_entropies = _generalized_entropy(scoring_rule, sample.samples.probabilities)  # (M, ...)
    return member_entropies.mean(0)
