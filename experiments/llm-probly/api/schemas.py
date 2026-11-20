"""Request and response models for the uncertainty API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UncertaintyRequest(BaseModel):
    """Body of ``POST /uncertainty``.

    Attributes:
        question: The question to answer.
        model: Short model name (e.g. ``qwen1.5b``, the file name in
            ``profiles/``) or Hugging Face id. A profile must exist for it.
        budget: Optional maximum relative cost per question (1 = greedy
            answer only, as in ``method.cost`` of the profile). The request
            is rejected if the model's chosen method costs more.
    """

    question: str = Field(min_length=1)
    model: str = Field(min_length=1)
    budget: float | None = Field(default=None, ge=0.0)


class UncertaintyResponse(BaseModel):
    """Body returned by ``POST /uncertainty``.

    Attributes:
        answer: The model's answer.
        confidence: Calibrated confidence in [0, 1].
        method: Name of the uncertainty method that was used.
        cost: Relative cost per question of that method, generation included.
    """

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    cost: float
