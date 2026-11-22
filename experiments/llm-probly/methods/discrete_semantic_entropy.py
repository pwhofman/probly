"""Plugin: discrete semantic entropy (clusters weighted by sample count) from the shared NLI matrix."""

from __future__ import annotations

from methods._se_common import score_rows

NAME = "discrete_semantic_entropy"


def score(rows: list[dict], model: object = None, tok: object = None) -> dict:
    """Score every row; see CONTRACT.md for the record format."""
    return score_rows(rows, NAME, which=1)
