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

# Fields of transformers ModelOutput subclasses that hold the model's prediction, in lookup order.
PREDICTION_FIELDS = ("logits", "predicted_depth", "prediction_outputs", "reconstruction")


def extract_predictions(output: Any) -> torch.Tensor:  # noqa: ANN401
    """Extract the prediction tensor from a transformers model output.

    Predictions are looked up under the known output fields in :data:`PREDICTION_FIELDS`, covering
    classification and regression heads (``logits``), depth estimation (``predicted_depth``), time series
    forecasting (``prediction_outputs``), and image reconstruction (``reconstruction``).

    Args:
        output: A transformers ``ModelOutput`` or a plain tensor.

    Returns:
        The prediction tensor.

    Raises:
        TypeError: If the output has none of the known prediction fields.
    """
    if isinstance(output, torch.Tensor):
        return output
    for field in PREDICTION_FIELDS:
        predictions = getattr(output, field, None)
        if predictions is not None:
            return predictions
    msg = (
        f"Cannot extract predictions from a {type(output).__name__}; probly requires a transformers model with "
        f"a prediction head whose output has one of the fields {PREDICTION_FIELDS} "
        "(config option return_dict=True, the default)."
    )
    raise TypeError(msg)


@predict_raw.register(PreTrainedModel)
def _predict_raw_transformers(predictor: PreTrainedModel, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
    """Call a transformers model and return its predictions."""
    return extract_predictions(predictor(*args, **kwargs))


__all__ = []
