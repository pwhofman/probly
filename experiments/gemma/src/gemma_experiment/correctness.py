"""NLI-based cluster correctness checking against ground truth."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma_experiment.entailment import EntailmentModel


def check_cluster_correctness(
    cluster_responses: list[str],
    ground_truth: str,
    entailment_model: EntailmentModel,
) -> bool:
    """Check if a semantic cluster is correct by comparing to ground truth.

    Takes the first response as a representative and checks bidirectional
    entailment against the ground-truth string. Uses the same equivalence
    criterion as ``get_semantic_ids``: neither direction is contradiction
    and the pair is not both neutral.

    Args:
        cluster_responses: All responses belonging to one semantic cluster.
        ground_truth: The expected correct answer.
        entailment_model: NLI model for entailment checking.
    """
    representative = cluster_responses[0]
    fwd = entailment_model.check_implication(representative, ground_truth)
    bwd = entailment_model.check_implication(ground_truth, representative)
    pair = [fwd, bwd]
    return (0 not in pair) and (pair != [1, 1])
