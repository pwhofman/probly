"""Evidential Deep Learning method for uncertainty quantification."""

from __future__ import annotations
from probly.representation.distribution.common import DirichletDistribution

from typing import TYPE_CHECKING, Unpack

from probly.predictor import predict
from probly.transformation.evidential.classification.common import evidential_classification
from probly.representation.distribution import

if TYPE_CHECKING:
    from probly.predictor import Predictor


class EvidentialClassification[In, KwIn, Out]:
    """Evidential classification for uncertainty quantification.

    Based on :cite:t:`sensoyEvidentialDeep2018`.
    """

    predictor: Predictor[In, KwIn, Out]

    def __init__(self, base: Predictor[In, KwIn, Out]) -> None:
        """Initialize EvidentialClassification.

        Args:
            base: Base model outputting raw logits (no Softmax).
                  Softplus is appended automatically.
        """
        self.predictor = evidential_classification(base)
        self.distribution = DirichletDistribution


    def predict(self, *args: In, **kwargs: Unpack[KwIn]) -> Out:
        """Run a single forward pass and return evidence values."""
        return predict(self.predictor, *args, **kwargs)
