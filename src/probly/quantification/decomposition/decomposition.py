"""Base class uncertainty decomposition methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, override

from probly.quantification._quantification import Quantifier
from probly.quantification.notion import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    Notion,
    NotionKey,
    NotionName,
    TotalUncertainty,
    notion_registry,
)
from probly.representation.representation import Representation


class Decomposition(Mapping[NotionKey, Notion], ABC):
    """Protocol for uncertainty decompositions."""

    canonical_notion: type[Notion] | None = None

    @property
    @abstractmethod
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return []

    @abstractmethod
    def _get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        msg = f"Notion {notion} is not a component of the decomposition."
        raise KeyError(msg)

    def get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        return self._get_notion(notion)

    def get_canonical(self) -> Notion:
        """Return the canonical notion of the decomposition."""
        if self.canonical_notion is None:
            msg = f"Decomposition of type {type(self)} does not have a canonical notion."
            raise NotImplementedError(msg)

        return self.get_notion(self.canonical_notion)

    @override
    def __getitem__[N: Notion](self, key: NotionName | type[N]) -> N:
        """Return the component corresponding to the given key."""
        notion_cls = key if isinstance(key, type) else notion_registry[key]
        if notion_cls not in self.components:
            msg = f"Notion {notion_cls} is not a component of the decomposition."
            raise KeyError(msg)
        return self.get_notion(notion_cls)

    @override
    def __iter__(self) -> Iterator[type[Notion]]:
        """Return an iterator over the keys of the components in the decomposition."""
        return iter(self.components)

    @override
    def __len__(self) -> int:
        """Return the number of components in the decomposition."""
        return len(self.components)


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class CachingDecomposition(Decomposition, ABC):
    """Protocol for decompositions that cache their components."""

    _cache: dict[type[Notion], Notion] = field(default_factory=dict, init=False, repr=False, compare=False)

    @override
    def get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion not in self._cache:
            self._cache[notion] = self._get_notion(notion)
        return self._cache[notion]  # ty:ignore[invalid-return-type]

    @override
    def __repr__(self) -> str:
        """Return a string representation of the decomposition."""
        components_str = ", ".join(f"{notion.__name__}={self.get_notion(notion)!r}" for notion in self.components)
        return f"{type(self).__name__}({components_str})"


class AleatoricDecomposition[AU: AleatoricUncertainty](Decomposition, ABC):  # ty:ignore[inconsistent-mro]
    """Protocol for decompositions with aleatoric uncertainty."""

    canonical_notion = AleatoricUncertainty

    @property
    @override
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return [AleatoricUncertainty, *super().components]

    @override
    def _get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion is AleatoricUncertainty:
            return self._aleatoric  # ty:ignore[invalid-return-type]
        return super()._get_notion(notion)

    @property
    @abstractmethod
    def _aleatoric(self) -> AU:
        """The aleatoric uncertainty of the decomposition."""

    @property
    def aleatoric(self) -> AU:
        """The aleatoric uncertainty of the decomposition."""
        return self.get_notion(AleatoricUncertainty)


class EpistemicDecomposition[EU: EpistemicUncertainty](Decomposition, ABC):  # ty:ignore[inconsistent-mro]
    """Protocol for decompositions with epistemic uncertainty."""

    canonical_notion = EpistemicUncertainty

    @property
    @override
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return [EpistemicUncertainty, *super().components]

    @override
    def _get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion is EpistemicUncertainty:
            return self._epistemic  # ty:ignore[invalid-return-type]
        return super()._get_notion(notion)

    @property
    @abstractmethod
    def _epistemic(self) -> EU:
        """The epistemic uncertainty of the decomposition."""

    @property
    def epistemic(self) -> EU:
        """The epistemic uncertainty of the decomposition."""
        return self.get_notion(EpistemicUncertainty)


class TotalDecomposition[TU: TotalUncertainty](Decomposition, ABC):  # ty:ignore[inconsistent-mro]
    """Protocol for decompositions with total uncertainty."""

    canonical_notion = TotalUncertainty

    @property
    @override
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return [TotalUncertainty, *super().components]

    @override
    def _get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion is TotalUncertainty:
            return self._total  # ty:ignore[invalid-return-type]
        return super()._get_notion(notion)

    @property
    @abstractmethod
    def _total(self) -> TU:
        """The total uncertainty of the decomposition."""

    @property
    def total(self) -> TU:
        """The total uncertainty of the decomposition."""
        return self.get_notion(TotalUncertainty)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ConstantTotalDecomposition[TU: TotalUncertainty](TotalDecomposition[TU]):
    """Protocol for decompositions where the total uncertainty is constant."""

    uncertainty: TU

    @property
    @override
    def _total(self) -> TU:
        """The total uncertainty of the decomposition."""
        return self.uncertainty


class AleatoricEpistemicDecomposition[AU: AleatoricUncertainty, EU: EpistemicUncertainty](
    AleatoricDecomposition[AU], EpistemicDecomposition[EU]
):
    """Protocol for decompositions into aleatoric and epistemic uncertainty."""

    # For AU/EU decompositions, there is no canonical notion, as AU and EU are equally valid notions of uncertainty.
    canonical_notion = None


class AleatoricTotalDecomposition[AU: AleatoricUncertainty, TU: TotalUncertainty](
    TotalDecomposition[TU], AleatoricDecomposition[AU]
):
    """Protocol for decompositions into aleatoric and total uncertainty.

    At least one of the two components (_aleatoric, _total) must be implemented,
    the other is then defined to be the same as the implemented component.
    """

    @override
    @property
    def _total(self) -> TU:
        """The total uncertainty of the decomposition."""
        return self.aleatoric  # ty:ignore[invalid-return-type]

    @override
    @property
    def _aleatoric(self) -> AU:
        """The aleatoric uncertainty of the decomposition."""
        return self.total  # ty:ignore[invalid-return-type]


class EpistemicTotalDecomposition[EU: EpistemicUncertainty, TU: TotalUncertainty](
    TotalDecomposition[TU], EpistemicDecomposition[EU]
):
    """Protocol for decompositions into epistemic and total uncertainty.

    At least one of the two components (_epistemic, _total) must be implemented,
    the other is then defined to be the same as the implemented component.
    """

    @override
    @property
    def _total(self) -> TU:
        """The total uncertainty of the decomposition."""
        return self.epistemic  # ty:ignore[invalid-return-type]

    @override
    @property
    def _epistemic(self) -> EU:
        """The epistemic uncertainty of the decomposition."""
        return self.total  # ty:ignore[invalid-return-type]


class AleatoricEpistemicTotalDecomposition[AU: AleatoricUncertainty, EU: EpistemicUncertainty, TU: TotalUncertainty](
    TotalDecomposition[TU], AleatoricDecomposition[AU], EpistemicDecomposition[EU]
):
    """Protocol for decompositions into aleatoric, epistemic and total uncertainty."""


class AdditiveDecomposition[AU: AleatoricUncertainty, EU: EpistemicUncertainty, TU: TotalUncertainty](
    AleatoricEpistemicTotalDecomposition[AU, EU, TU], CachingDecomposition
):
    """Protocol for decompositions where AU and EU sum up to the total uncertainty.

    At least two of the three components (_total, _aleatoric, _epistemic) must be implemented,
    the third is then computed as the difference or sum of the other components.
    """

    @override
    @property
    def _total(self) -> TU:
        """The total uncertainty of the decomposition."""
        return self.aleatoric + self.epistemic  # ty:ignore[unsupported-operator]

    @override
    @property
    def _aleatoric(self) -> AU:
        """The aleatoric uncertainty of the decomposition."""
        return self.total - self.epistemic  # ty:ignore[unsupported-operator]

    @override
    @property
    def _epistemic(self) -> EU:
        """The epistemic uncertainty of the decomposition."""
        return self.total - self.aleatoric  # ty:ignore[unsupported-operator]


class Decomposer[R: Representation, D: Decomposition](Quantifier[R, D], Protocol):
    """Protocol for uncertainty quantification methods that also decompose the uncertainty."""
