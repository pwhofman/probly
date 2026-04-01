"""Generic representation builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch

if TYPE_CHECKING:
    from probly.predictor import Predictor


class Representer[**CtrIn, **In, Out](ABC):
    """Abstract base class for representation builders."""

    predictor: Predictor[In, Out]

    def __init__(
        self,
        predictor: Predictor[In, Out],
        *_args: CtrIn.args,
        **_kwargs: CtrIn.kwargs,
    ) -> None:
        """Initialize the representer with a predictor.

        Args:
            predictor: The predictor to be used for building representations.

        """
        self.predictor = predictor

    @abstractmethod
    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Build a representation for a given input."""
        raise NotImplementedError

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Predict the representation for a given input."""
        return self(*args, **kwargs)


@lazydispatch
def representer[**CtrIn, **In, Out](predictor: Predictor[In, Out]) -> Representer[CtrIn, In, Out]:
    """Generic represent function."""
    msg = f"No represent function registered for type {type(predictor)}"
    raise NotImplementedError(msg)
