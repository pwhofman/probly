"""Bindings for Hugging Face transformers predictors.

Transformers models return ``ModelOutput`` dataclasses instead of tensors. Registering ``predict_raw`` for
``PreTrainedModel`` unwraps the ``logits`` field, so transformers classifiers plug into the prediction,
representation, and quantification pipeline like any other torch module, including after transformations
such as ``dropout`` or ``ensemble``.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel

from probly.predictor import predict_raw


def extract_logits(output: Any) -> torch.Tensor:  # noqa: ANN401
    """Extract the logits tensor from a transformers model output.

    Args:
        output: A transformers ``ModelOutput`` or a plain tensor.

    Returns:
        The logits tensor.

    Raises:
        TypeError: If the output has no ``logits`` field.
    """
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    msg = (
        f"Cannot extract logits from a {type(output).__name__}; probly requires a transformers model with a "
        "prediction head (e.g. a *ForSequenceClassification or *ForImageClassification model) that returns a "
        "ModelOutput with a 'logits' field (config option return_dict=True, the default)."
    )
    raise TypeError(msg)


@predict_raw.register(PreTrainedModel)
def _predict_raw_transformers(predictor: PreTrainedModel, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
    """Call a transformers model and return its logits."""
    return extract_logits(predictor(*args, **kwargs))


__all__ = []
