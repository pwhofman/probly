"""Base class uncertainty decomposition methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import override

from probly.quantification._quantification import Quantification, Quantifier
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


class Decomposition(Quantification, Mapping[NotionKey, Notion], ABC):
    """Protocol for uncertainty decompositions."""

    @property
    @abstractmethod
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        raise NotImplementedError

    @abstractmethod
    def _get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        raise NotImplementedError

    def get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        return self._get_notion(notion)

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


class CachingDecomposition(Decomposition, ABC):
    """Protocol for decompositions that cache their components."""

    _caching: bool
    _cache: dict[type[Notion], Notion]

    def __init__(self, caching: bool = True) -> None:
        """Initialize the cache."""
        self._caching = caching
        self._cache = {}

    @override
    def get_notion[N: Notion](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if self._caching and notion not in self._cache:
            self._cache[notion] = self._get_notion(notion)
        return self._cache[notion]  # ty:ignore[invalid-return-type]


class AleatoricEpistemicDecomposition(Decomposition, ABC):
    """Protocol for decompositions into aleatoric and epistemic uncertainty."""

    @property
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return [AleatoricUncertainty, EpistemicUncertainty]

    @override
    def _get_notion[N: (AleatoricUncertainty, EpistemicUncertainty)](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion is AleatoricUncertainty:
            return self._aleatoric
        if notion is EpistemicUncertainty:
            return self.epistemic
        msg = f"Notion {notion} is not a component of the decomposition."
        raise KeyError(msg)

    @property
    @abstractmethod
    def _aleatoric(self) -> AleatoricUncertainty:
        """The aleatoric uncertainty of the decomposition."""

    @property
    @abstractmethod
    def _epistemic(self) -> EpistemicUncertainty:
        """The epistemic uncertainty of the decomposition."""

    @property
    def aleatoric(self) -> AleatoricUncertainty:
        """The aleatoric uncertainty of the decomposition."""
        return self.get_notion(AleatoricUncertainty)

    @property
    def epistemic(self) -> EpistemicUncertainty:
        """The epistemic uncertainty of the decomposition."""
        return self.get_notion(EpistemicUncertainty)


class AleatoricEpistemicTotalDecomposition(AleatoricEpistemicDecomposition, ABC):
    """Protocol for decompositions into aleatoric, epistemic and total uncertainty."""

    @property
    def components(self) -> list[type[Notion]]:
        """The components of the decomposition."""
        return [AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty]

    @override
    def _get_notion[N: (AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty)](self, notion: type[N]) -> N:
        """Return the component corresponding to the given notion."""
        if notion is TotalUncertainty:
            return self._total
        return super()._get_notion(notion)

    @property
    @abstractmethod
    def _total(self) -> TotalUncertainty:
        """The total uncertainty of the decomposition."""

    @property
    def total(self) -> TotalUncertainty:
        """The total uncertainty of the decomposition."""
        return self.get_notion(TotalUncertainty)


class AdditiveDecomposition(AleatoricEpistemicTotalDecomposition, CachingDecomposition, ABC):
    """Protocol for decompositions where AU and EU sum up to the total uncertainty.

    At least two of the three components (_total, _aleatoric, _epistemic) must be implemented,
    the third is then computed as the difference or sum of the other components.
    """

    @override
    @property
    def _total(self) -> TotalUncertainty:
        """The total uncertainty of the decomposition."""
        return self.aleatoric + self.epistemic  # ty:ignore[unsupported-operator]

    @override
    @property
    def _aleatoric(self) -> AleatoricUncertainty:
        """The aleatoric uncertainty of the decomposition."""
        return self.total - self.epistemic  # ty:ignore[unsupported-operator]

    @override
    @property
    def _epistemic(self) -> EpistemicUncertainty:
        """The epistemic uncertainty of the decomposition."""
        return self.total - self.aleatoric  # ty:ignore[unsupported-operator]


class DecomposingQuantifier[R: Representation, D: Decomposition](Quantifier[R, D], ABC):
    """Protocol for uncertainty quantification methods that also decompose the uncertainty."""

    @abstractmethod
    def decompose(self, representation: R) -> D:
        """Decompose the uncertainty of a given representation."""

    def __call__(self, representation: R) -> D:
        """Quantify the uncertainty of a given representation."""
        return self.decompose(representation)
