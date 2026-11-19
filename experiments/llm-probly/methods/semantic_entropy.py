"""Plugin: semantic entropy (clusters weighted by sample likelihood) from the shared NLI matrix."""

from __future__ import annotations

from methods._se_common import score_rows

NAME = "semantic_entropy"
FAMILY = "agreement"
NEEDS_MODEL = False


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:  # noqa: ARG001
    """Score every row; see CONTRACT.md for the record format."""
    return score_rows(rows, NAME, which=0)
