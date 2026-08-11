"""Bindings for PEFT (LoRA and other adapter) predictors.

PEFT wraps transformers models in a ``PeftModel`` that is not a ``PreTrainedModel`` subclass, so it misses
the transformers binding; registering ``predict_raw`` for it unwraps predictions the same way.
"""

from __future__ import annotations

from typing import Any

import torch
from peft.peft_model import PeftModel

from probly.predictor import predict_raw

from .transformers import extract_predictions


@predict_raw.register(PeftModel)
def _predict_raw_peft(predictor: PeftModel, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
    """Call a PEFT-wrapped transformers model and return its predictions."""
    return extract_predictions(predictor(*args, **kwargs))


__all__ = []
