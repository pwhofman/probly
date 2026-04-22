"""Common abstractions for probability distributions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from flextype import flexdispatch
from probly.representation.representation import Representation
from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike

type DistributionType = Literal["gaussian", "dirichlet", "categorical", "empirical_second_order_categorical"]


class Distribution[T](Representation, ABC):
    """Base class for distributions."""

    type: DistributionType

    def entropy(self) -> ArrayLike:
        """Compute entropy."""
        from probly.quantification.measure.distribution._common import entropy  # noqa: PLC0415

        return entropy(self)

    @abstractmethod
    def sample(self, num_samples: int) -> Sample[T]:
        """Draw samples from Distribution."""


class CategoricalDistribution[T](Distribution[T]):
    """Base class for categorical distributions."""

    type: Literal["categorical"] = "categorical"

    @property
    @abstractmethod
    def unnormalized_probabilities(self) -> ArrayLike:
        """Get the probabilities of the categorical distribution."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get the number of classes."""

    @property
    @abstractmethod
    def probabilities(self) -> ArrayLike:
        """Get the normalized probabilities of the categorical distribution."""


class DistributionSample[T: Distribution](Sample[T]):
    """Sample type for empirical second-order distributions."""

    _running_instancehook: ClassVar[ContextVar[object]] = ContextVar(
        "DistributionSample._running_instancehook", default=NotImplemented
    )
    sample_space: ClassVar[type[Distribution]] = Distribution

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        if cls._running_instancehook.get() is instance:
            return NotImplemented
        try:
            tok = cls._running_instancehook.set(instance)
            if isinstance(instance, Sample) and isinstance(instance.samples, cls.sample_space):
                return True
        finally:
            cls._running_instancehook.reset(tok)
        return NotImplemented


class CategoricalDistributionSample[T: CategoricalDistribution](DistributionSample[T]):
    """Sample type for empirical second-order categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = CategoricalDistribution

    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


class SecondOrderDistribution[T: Distribution](Distribution[T]):
    """Base class for second-order distributions."""


class DirichletDistribution[T: CategoricalDistribution](SecondOrderDistribution[T]):
    """Base class for Dirichlet distributions."""

    type: Literal["dirichlet"] = "dirichlet"

    @property
    @abstractmethod
    def alphas(self) -> Any:  # noqa: ANN401
        """Get the concentration parameters of the Dirichlet distribution."""


class GaussianDistribution[D](Distribution[D], ABC):
    """Base class for Gaussian distributions."""

    type: Literal["gaussian"] = "gaussian"

    @property
    @abstractmethod
    def mean(self) -> D:
        """Get the mean parameters."""

    @property
    @abstractmethod
    def var(self) -> D:
        """Get the var parameters."""


@flexdispatch
def create_categorical_distribution[T](data: T) -> CategoricalDistribution:
    """Create a categorical distribution from backend-specific probability data."""
    msg = f"No categorical distribution factory registered for data type {type(data)}"
    raise NotImplementedError(msg)


@create_categorical_distribution.register(CategoricalDistribution)
def _(data: CategoricalDistribution) -> CategoricalDistribution:
    """Create a categorical distribution from an instance of CategoricalDistribution."""
    return data


@flexdispatch
def create_categorical_distribution_from_logits[T](data: T) -> CategoricalDistribution:
    """Create a categorical distribution from backend-specific logit data."""
    msg = f"No categorical distribution factory from logits registered for data type {type(data)}"
    raise NotImplementedError(msg)


@flexdispatch
def create_dirichlet_distribution_from_alphas[T](alphas: T) -> DirichletDistribution:
    """Create a Dirichlet distribution from backend-specific alpha data."""
    msg = f"No Dirichlet distribution factory registered for alphas type {type(alphas)}"
    raise NotImplementedError(msg)
