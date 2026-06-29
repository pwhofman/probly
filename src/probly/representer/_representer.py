"""Generic representation builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, final, override

from flextype import flexdispatch
from probly.predictor import RepresentationPredictor, predict
from probly.representation import Representation

if TYPE_CHECKING:
    from probly.predictor import Predictor


class Representer[**CtrIn, **In, Out, R: Representation](RepresentationPredictor[In, R], ABC):
    """Abstract base class for representation builders.

    A representer is a configurable representation predictor which wraps a base predictor
    and builds a representation for its predictions. The base predictor is passed to the representer
    during initialization and can be used to build the representation in the `represent` method.
    """

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
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> R:
        """Build a representation for a given input."""
        raise NotImplementedError

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> R:
        """Alias for the represent method."""
        return self.represent(*args, **kwargs)

    def predict(self, *args: In.args, **kwargs: In.kwargs) -> R:
        """Predict the representation for a given input."""
        return self.represent(*args, **kwargs)


@final
class DummyRepresenter[**In, Out: Representation](Representer[Any, In, Out, Out]):
    """A dummy representer that simply calls the base predictor."""

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Build a representation for a given input."""
        return predict(self.predictor, *args, **kwargs)


@flexdispatch
def representer[**CtrIn, **In, Out, R: Representation](
    predictor: Predictor[In, Out], *args: CtrIn.args, **kwargs: CtrIn.kwargs
) -> Representer[CtrIn, In, Out, R]:
    """Generic represent function."""
    msg = f"No represent function registered for type {type(predictor)}"
    raise NotImplementedError(msg)


@representer.register(RepresentationPredictor)
def represent_representation_predictor[**In, Out: Representation](
    predictor: RepresentationPredictor[In, Out],
) -> DummyRepresenter[In, Out]:
    """Predictors that are already representation predictors can be called directly."""
    return DummyRepresenter(predictor)
