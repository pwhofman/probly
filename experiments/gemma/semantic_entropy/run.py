"""Compute semantic entropy over multiple Gemma 4 responses to the same question.

Semantic entropy clusters LLM responses by meaning using an NLI model
(bidirectional entailment) and measures the Shannon entropy of the cluster
frequency distribution. High entropy indicates the model gives semantically
diverse answers (uncertain); low entropy indicates consistent meaning (confident).

Two variants are computed:
  - Discrete: cluster probabilities from uniform sample counts
  - Weighted: cluster probabilities from generation log-likelihoods

Usage:
    uv run python gemma/semantic_entropy/run.py --num-samples 10 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core import (
    CACHE_DIR,
    DEFAULT_NLI_MODEL,
    EntailmentModel,
    cluster_assignment_entropy,
    generate_responses,
    get_semantic_ids,
    suppress_hf_noise,
    weighted_semantic_entropy,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma import MODEL_ID

QUESTIONS: list[tuple[str, str]] = [
    ("Factual", "What is the capital of France?"),
    ("Factual", "What is the chemical symbol for water?"),
    ("Factual", "How many planets are in our solar system?"),
    ("Explanation", "Why is the sky blue?"),
    ("Explanation", "What causes tides in the ocean?"),
    ("Subjective", "What is the best programming language?"),
    ("Subjective", "Is a hot dog a sandwich?"),
    ("Trick", "Who was the first person to walk on Mars?"),
    ("Trick", "What year was the city of Atlantis founded?"),
]


@dataclass
class QuestionResult:
    """Results of semantic entropy computation for a single question."""

    category: str
    question: str
    responses: list[str]
    log_likelihoods: list[float]
    semantic_ids: list[int]
    entropy_discrete: float
    entropy_weighted: float

    @property
    def num_clusters(self) -> int:
        """Return the number of distinct semantic clusters."""
        return len(set(self.semantic_ids))


def format_result(result: QuestionResult, index: int, total: int) -> str:
    """Format a single question's results for printing."""
    lines = [
        f"=== Question {index}/{total} [{result.category}] ===",
        f"Q: {result.question}",
        "",
        f"Responses ({len(result.responses)} samples, {result.num_clusters} clusters):",
    ]

    clusters: dict[int, list[tuple[int, str, float]]] = {}
    for i, (sid, resp, ll) in enumerate(
        zip(result.semantic_ids, result.responses, result.log_likelihoods, strict=True)
    ):
        clusters.setdefault(sid, []).append((i + 1, resp, ll))

    for cid in sorted(clusters):
        members = clusters[cid]
        lines.append(f"  [Cluster {cid}] ({len(members)} responses)")
        for idx, text, ll in members:
            display = text[:180] + "..." if len(text) > 180 else text
            lines.append(f"    {idx} (ll={ll:.3f}): {display!r}")

    lines.append("")
    lines.append(f"Semantic Entropy (discrete): {result.entropy_discrete:.4f}")
    lines.append(f"Semantic Entropy (weighted): {result.entropy_weighted:.4f}")
    return "\n".join(lines)


def print_summary(results: list[QuestionResult]) -> None:
    """Print a compact summary table."""
    print("\n" + "=" * 78)
    print("Summary")
    print("-" * 78)
    print(f"{'Question':<40} {'Clusters':>8} {'Discrete':>10} {'Weighted':>10}")
    print("-" * 78)
    for r in results:
        q = r.question[:38] + ".." if len(r.question) > 40 else r.question
        print(f"{q:<40} {r.num_clusters:>8} {r.entropy_discrete:>10.4f} {r.entropy_weighted:>10.4f}")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of response samples per question (default: 10).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Gemma (default: 0.7).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens per generated response (default: 128).",
    )
    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL,
        help="NLI model for entailment checking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    suppress_hf_noise()

    print(f"Loading Gemma model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
    model.eval()

    print(f"Loading NLI model: {args.nli_model}")
    entailment = EntailmentModel(args.nli_model)

    print(
        f"\nRunning semantic entropy: {len(QUESTIONS)} questions, "
        f"{args.num_samples} samples each, temperature={args.temperature}\n"
    )

    results: list[QuestionResult] = []
    for i, (category, question) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Generating responses for: {question}")
        responses, log_likelihoods = generate_responses(
            question,
            model,
            tokenizer,
            args.num_samples,
            args.temperature,
            args.max_new_tokens,
        )
        print(f"  Clustering {len(responses)} responses...")
        semantic_ids = get_semantic_ids(responses, entailment)
        se_discrete = cluster_assignment_entropy(semantic_ids)
        se_weighted = weighted_semantic_entropy(semantic_ids, log_likelihoods)

        result = QuestionResult(
            category=category,
            question=question,
            responses=responses,
            log_likelihoods=log_likelihoods,
            semantic_ids=semantic_ids,
            entropy_discrete=se_discrete,
            entropy_weighted=se_weighted,
        )
        results.append(result)
        print(format_result(result, i, len(QUESTIONS)))
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
