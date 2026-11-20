"""FastAPI proof of concept for calibrated uncertainty."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from selector.profile import confidence

from .profiles import find_profile
from .schemas import UncertaintyRequest, UncertaintyResponse
from .scoring import SCORERS, generate

app = FastAPI(
    title="llm-probly uncertainty API",
    description="Answer a question and return a calibrated confidence, using the model's saved profile.",
)


@app.post("/uncertainty", response_model=UncertaintyResponse)
def uncertainty(req: UncertaintyRequest) -> UncertaintyResponse:
    """Answer a question and return a calibrated confidence.

    Args:
        req: Question, model (short name or Hugging Face id) and optional cost budget.

    Returns:
        Answer, calibrated confidence, chosen method and its cost.

    Raises:
        HTTPException: 404 if no profile exists for the model, 422 if the chosen
            method costs more than the budget, 501 if the API cannot compute the
            chosen method yet, 500 if the profile file is invalid or cannot
            calibrate the score.
    """
    try:
        profile = find_profile(req.model)
    except ValueError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for model: {req.model}")

    method = profile.method
    if req.budget is not None and method.cost > req.budget:
        raise HTTPException(
            status_code=422,
            detail=f"Method '{method.name}' costs {method.cost:g}, above budget {req.budget:g}.",
        )
    scorer = SCORERS.get(method.name)
    if scorer is None:
        raise HTTPException(
            status_code=501,
            detail=f"The API cannot compute '{method.name}' yet. Supported: {', '.join(sorted(SCORERS))}.",
        )

    gen = generate(profile.model_id, req.question)
    try:
        conf = float(confidence(profile, scorer(gen)))
    except ValueError as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    return UncertaintyResponse(
        answer=gen.answer,
        confidence=min(1.0, max(0.0, conf)),
        method=method.name,
        cost=method.cost,
    )
