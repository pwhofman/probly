"""Shared DDU representer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import torch

from probly.method.ddu import DDUPredictor
from probly.predictor import predict
from probly.representation.ddu import DDURepresentation
from probly.representation.ddu.torch import TorchDDURepresentation
from probly.representation.distribution.torch_categorical import TorchTensorCategoricalDistribution
from probly.representer._representer import Representer, representer

if TYPE_CHECKING:
    from collections.abc import Iterable


@representer.register(DDUPredictor)
class DDURepresenter[**In, Out](Representer[Any, In, Out, DDURepresentation]):
    """Representer for DDU predictors.

    Extracts softmax probabilities and penultimate-layer features from a DDU
    model in a single forward pass. The softmax entropy measures aleatoric
    uncertainty; the features are used to fit a density model for epistemic
    uncertainty.
    """

    def _predict(self, *args: In.args, **kwargs: In.kwargs) -> Iterable[Out]:
        """Predict multiple outputs from the ensemble predictor."""
        return predict(self.predictor, *args, **kwargs)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> TorchDDURepresentation:
        """Extract softmax and features from the DDU predictor.

        Args:
            *args: Positional arguments forwarded to the predictor.
            **kwargs: Keyword arguments forwarded to the predictor.

        Returns:
            DDU representation containing softmax probabilities and feature vectors.
        """
        features: torch.Tensor = self.predictor(*args, **kwargs)
        logits: torch.Tensor = self.predictor.classification_head(features)
        return TorchDDURepresentation(
            softmax=TorchTensorCategoricalDistribution(probabilities=torch.softmax(logits, dim=-1)),
            features=features,
        )
