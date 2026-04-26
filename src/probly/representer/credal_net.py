"""Representer for the credal net method based on :cite:`wang2024credalnet`."""

from __future__ import annotations

from typing import Any, override

from probly.method.credal_net import CredalNetPredictor
from probly.predictor import predict_raw
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representer._representer import Representer, representer


@representer.register(CredalNetPredictor)
class CredalNetRepresenter[**In, Out, C: ProbabilityIntervalsCredalSet](Representer[Any, In, Out, C]):
    """Representer that exposes the credal-set view of a credal net's interval output.

    Calls ``predict_raw`` (which packs the input and runs the model) to get
    the packed ``(B, 2C)`` interval tensor, then wraps it in a
    :class:`ProbabilityIntervalsCredalSet` (with reachability already enforced
    by the ``IntSoftmax`` head).
    """

    def __init__(self, predictor: CredalNetPredictor) -> None:
        """Initialize the representer with a credal net predictor."""
        super().__init__(predictor)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Run the model and wrap its packed interval output as a credal set."""
        raw = predict_raw(self.predictor, *args, **kwargs)
        return create_probability_intervals_from_lower_upper_array(raw)  # ty:ignore[invalid-return-type]
