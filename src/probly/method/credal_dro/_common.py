"""Credal DRO method compatibility layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.predictor import LogitClassifier, ProbabilisticClassifier
from probly.representer import ProbabilityIntervalsRepresenter, representer
from probly.transformation.ensemble import EnsemblePredictor, ensemble
from probly.transformation.transformation import predictor_transformation

if TYPE_CHECKING:
    from probly.predictor import Predictor


class CredalDROPredictor[**In, Out](EnsemblePredictor[In, Out], Protocol):
    """A predictor routed through the credal DRO representer."""


@predictor_transformation(
    permitted_predictor_types=(ProbabilisticClassifier, LogitClassifier), preserve_predictor_type=False
)
@CredalDROPredictor.register_factory(autocast_builtins=True)
def credal_dro[**In, Out](
    base: Predictor[In, Out], num_members: int, reset_params: bool = True
) -> CredalDROPredictor[In, Out]:
    """Create a credal DRO predictor from a base predictor based on :cite:`wangLearningCredalEnsembles2026`.

    Structurally identical to ``credal_wrapper``; the methods differ only in training.
    Member ``i`` is trained on the CVaR cross-entropy at level ``credal_dro_deltas(delta_g,
    num_members)[i]`` (see ``probly.train.credal.torch.cvar_ce_loss``).

    Args:
        base: The base classifier to replicate into an ensemble.
        num_members: Number of ensemble members.
        reset_params: Whether to reset the parameters of each member. Default is True.

    Returns:
        The credal DRO predictor outputting a ProbabilityIntervalsCredalSet.
    """
    return ensemble(base, num_members=num_members, reset_params=reset_params)


def credal_dro_deltas(delta_g: float, num_members: int) -> list[float]:
    """Per-member CVaR levels: uniform interpolation over ``[delta_g, 1]``.

    Implements Eq. 8 of :cite:`wangLearningCredalEnsembles2026`; the member at level 1
    is a plain ERM model. Note that the paper's released reference implementation instead
    interpolates over ``(delta_g, 1]``, excluding ``delta_g`` itself; we follow the paper.

    Args:
        delta_g: Global worst-case level in (0, 1]; the paper recommends [0.5, 1).
        num_members: Ensemble size.

    Returns:
        ``num_members`` levels, ``delta_g`` first and 1.0 last (a single member gets
        ``delta_g``).

    Raises:
        ValueError: If delta_g is outside (0, 1] or num_members < 1.
    """
    if not 0.0 < delta_g <= 1.0:
        msg = f"delta_g must be in (0, 1], got {delta_g}."
        raise ValueError(msg)
    if num_members < 1:
        msg = f"num_members must be >= 1, got {num_members}."
        raise ValueError(msg)
    if num_members == 1:
        return [delta_g]
    # Endpoints are pinned exactly: float rounding in delta_g + (1 - delta_g) can land just below
    # 1.0, which would deny the ERM member the delta == 1 shortcut of the CVaR loss.
    interior = [delta_g + (1.0 - delta_g) * i / (num_members - 1) for i in range(1, num_members - 1)]
    return [delta_g, *interior, 1.0]


representer.register(CredalDROPredictor, ProbabilityIntervalsRepresenter)

__all__ = ["CredalDROPredictor", "credal_dro", "credal_dro_deltas"]
