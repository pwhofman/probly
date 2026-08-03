"""Compare semantic entropy and spectral uncertainty on Gemma 4 responses.

Runs both methods side-by-side on the same questions and samples:
  - Semantic entropy: NLI-based clustering + Shannon entropy (Kuhn et al., 2024)
  - Spectral uncertainty: sentence embeddings + RBF kernel + Von Neumann entropy
    (Walha et al., 2025), with aleatoric/epistemic decomposition via clarifications

Usage:
    uv run python experiments/spectral_uncertainty.py --num-samples 10 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from gemma_experiment import (
    CACHE_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_NLI_MODEL,
    MODEL_ID,
    EntailmentModel,
    SentenceEmbedder,
    SpectralDecomposition,
    cluster_assignment_entropy,
    generate_clarifications,
    generate_responses,
    get_semantic_ids,
    spectral_decomposed_uncertainty,
    spectral_total_uncertainty,
    suppress_hf_noise,
    weighted_semantic_entropy,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Results of both uncertainty methods for a single question."""

    category: str
    question: str
    responses: list[str]
    log_likelihoods: list[float]
    # Semantic entropy
    semantic_ids: list[int]
    entropy_discrete: float
    entropy_weighted: float
    # Spectral uncertainty
    spectral_total: float
    spectral_decomposition: SpectralDecomposition | None
    clarifications: list[str] | None

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
    ]

    if result.clarifications:
        lines.append(f"Clarifications ({len(result.clarifications)}):")
        for i, c in enumerate(result.clarifications, 1):
            display = c[:120] + "..." if len(c) > 120 else c
            lines.append(f"  {i}. {display!r}")
        lines.append("")

    lines.append(f"Responses ({len(result.responses)} samples, {result.num_clusters} clusters):")

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
    lines.append(f"Spectral Total Uncertainty:  {result.spectral_total:.4f}")

    if result.spectral_decomposition:
        d = result.spectral_decomposition
        lines.append(f"  Aleatoric:  {d.aleatoric:.4f}")
        lines.append(f"  Epistemic:  {d.epistemic:.4f}")

    return "\n".join(lines)


def print_summary(results: list[QuestionResult]) -> None:
    """Print a compact summary table."""
    has_decomposition = any(r.spectral_decomposition for r in results)

    print("\n" + "=" * 100)
    print("Summary")
    print("-" * 100)

    header = f"{'Question':<40} {'Clust':>5} {'SE-Disc':>8} {'SE-Wgt':>8} {'Spec-Tot':>9}"
    if has_decomposition:
        header += f" {'Aleat':>8} {'Epist':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        q = r.question[:38] + ".." if len(r.question) > 40 else r.question
        line = (
            f"{q:<40} {r.num_clusters:>5} {r.entropy_discrete:>8.4f}"
            f" {r.entropy_weighted:>8.4f} {r.spectral_total:>9.4f}"
        )
        if has_decomposition and r.spectral_decomposition:
            d = r.spectral_decomposition
            line += f" {d.aleatoric:>8.4f} {d.epistemic:>8.4f}"
        print(line)

    print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of response samples per question or per clarification (default: 10).",
    )
    parser.add_argument(
        "--num-clarifications",
        type=int,
        default=5,
        help="Number of clarifications per question for decomposition (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens per generated response (default: 128).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="RBF kernel bandwidth parameter (default: 1.0).",
    )
    parser.add_argument(
        "--nli-model",
        default=DEFAULT_NLI_MODEL,
        help="NLI model for entailment checking.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer model for embeddings.",
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

    print(f"Loading embedding model: {args.embed_model}")
    embedder = SentenceEmbedder(args.embed_model)

    print(
        f"\nRunning comparison: {len(QUESTIONS)} questions, "
        f"{args.num_clarifications} clarifications x {args.num_samples} samples each, "
        f"temperature={args.temperature}, gamma={args.gamma}\n"
    )

    results: list[QuestionResult] = []
    for i, (category, question) in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")

        # Stage 1: Generate clarifications
        print(f"  Generating {args.num_clarifications} clarifications...")
        clarifications = generate_clarifications(
            question, model, tokenizer, args.num_clarifications, args.temperature, args.max_new_tokens
        )

        # Stage 2: Generate responses per clarification
        all_responses: list[str] = []
        all_log_likelihoods: list[float] = []
        group_sizes: list[int] = []

        for ci, clarification in enumerate(clarifications, 1):
            print(f"  Clarification {ci}/{len(clarifications)}: generating {args.num_samples} responses...")
            responses, log_likelihoods = generate_responses(
                clarification, model, tokenizer, args.num_samples, args.temperature, args.max_new_tokens
            )
            all_responses.extend(responses)
            all_log_likelihoods.extend(log_likelihoods)
            group_sizes.append(len(responses))

        # Semantic entropy (on all responses)
        print(f"  Clustering {len(all_responses)} responses via NLI...")
        semantic_ids = get_semantic_ids(all_responses, entailment)
        se_discrete = cluster_assignment_entropy(semantic_ids)
        se_weighted = weighted_semantic_entropy(semantic_ids, all_log_likelihoods)

        # Spectral uncertainty
        print("  Computing spectral uncertainty...")
        embeddings = embedder.embed(all_responses)
        spec_total = spectral_total_uncertainty(embeddings, args.gamma)
        spec_decomp = spectral_decomposed_uncertainty(embeddings, group_sizes, args.gamma)

        result = QuestionResult(
            category=category,
            question=question,
            responses=all_responses,
            log_likelihoods=all_log_likelihoods,
            semantic_ids=semantic_ids,
            entropy_discrete=se_discrete,
            entropy_weighted=se_weighted,
            spectral_total=spec_total,
            spectral_decomposition=spec_decomp,
            clarifications=clarifications,
        )
        results.append(result)
        print(format_result(result, i, len(QUESTIONS)))
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
