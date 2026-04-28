"""Representer for the efficient credal prediction method based on :cite:`hofmanefficient`."""

from __future__ import annotations

from typing import Any, override

import torch

from probly.method.efficient_credal_prediction import EfficientCredalPredictor
from probly.predictor import predict_raw
from probly.representation.credal_set import (
    ProbabilityIntervalsCredalSet,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representation.distribution import CategoricalDistribution
from probly.representer._representer import Representer, representer


@representer.register(EfficientCredalPredictor)
class EfficientCredalRepresenter[**In, Out: CategoricalDistribution, C: ProbabilityIntervalsCredalSet](
    Representer[Any, In, Out, C]
):
    """Builds a credal set from the base logits and the calibrated logit-space offsets.

    For each class ``k``, the kth logit is perturbed by ``lower[k]`` and
    ``upper[k]`` (signed: ``lower`` is non-positive, ``upper`` is non-negative)
    independently of the others, and the result is softmaxed. The credal set's
    ``i``th lower (resp. upper) bound is the min (resp. max) of the ``i``th
    coordinate across the 2K resulting distributions.
    """

    predictor: EfficientCredalPredictor[In, Out]

    def __init__(self, predictor: EfficientCredalPredictor[In, Out]) -> None:
        """Initialize the representer with an efficient credal predictor."""
        super().__init__(predictor)

    @override
    def represent(self, *args: In.args, **kwargs: In.kwargs) -> C:
        """Run the base, perturb each logit by the calibrated offsets, and reduce to credal bounds."""
        logits: torch.Tensor = predict_raw(self.predictor, *args, **kwargs)
        n_classes = logits.shape[-1]
        lower: torch.Tensor = self.predictor.lower
        upper: torch.Tensor = self.predictor.upper

        # Per-class perturbations as diagonal matrices: row k has the kth-position
        # offset and zeros elsewhere. Broadcasts over the batch dim cleanly.
        eye = torch.eye(n_classes, device=logits.device, dtype=logits.dtype)
        perturb_up = upper.unsqueeze(-1) * eye
        perturb_lo = lower.unsqueeze(-1) * eye

        # 2K perturbed logit tensors per input. Broadcasting creates a fresh
        # (B, K, K) tensor for each side -- no in-place mutation across iterations.
        logits_up = logits.unsqueeze(1) + perturb_up.unsqueeze(0)
        logits_lo = logits.unsqueeze(1) + perturb_lo.unsqueeze(0)

        probs_up = torch.softmax(logits_up, dim=-1)
        probs_lo = torch.softmax(logits_lo, dim=-1)

        # Per-output-class min/max across all 2K perturbations.
        probs_all = torch.cat([probs_up, probs_lo], dim=1)
        lower_bounds = probs_all.min(dim=1).values
        upper_bounds = probs_all.max(dim=1).values

        packed = torch.cat([lower_bounds, upper_bounds], dim=-1)
        return create_probability_intervals_from_lower_upper_array(packed)  # ty:ignore[invalid-argument-type, invalid-return-type]
