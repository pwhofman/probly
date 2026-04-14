"""Measure and improve semantic calibration on Gemma 4 responses.

Generates multiple responses per question, clusters them semantically,
checks correctness against ground truth via NLI, and evaluates calibration.
Both discrete (count-based) and weighted (log-likelihood) confidence variants
are reported.  Post-hoc calibrators are evaluated via leave-one-out CV.

Reference: arXiv:2511.04869 — "Semantic Calibration"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

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
from core.calibration import (
    CALIBRATORS,
    average_calibration_error,
    compute_semantic_confidence_discrete,
    compute_semantic_confidence_weighted,
    expected_calibration_error,
    leave_one_out_evaluate,
)
from core.correctness import check_cluster_correctness
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma import MODEL_ID

QUESTIONS: list[tuple[str, str, str | None]] = [
    # Easy factual (expect high confidence, correct) -- sanity baseline
    ("Factual", "What is the capital of France?", "The capital of France is Paris"),
    ("Factual", "What is the chemical symbol for water?", "The chemical symbol for water is H2O"),
    # Hard factual (model may split or hallucinate across samples)
    (
        "Hard",
        "What is the approximate population of Iceland?",
        "Iceland has a population of approximately 380,000 people",
    ),
    (
        "Hard",
        "Who was the second person to walk on the Moon?",
        "Buzz Aldrin was the second person to walk on the Moon",
    ),
    (
        "Hard",
        "In what year was the University of Bologna founded?",
        "The University of Bologna was founded in 1088",
    ),
    (
        "Hard",
        "What is the deepest point in the ocean?",
        "The Challenger Deep in the Mariana Trench is the deepest point",
    ),
    # Ambiguous (debatable, expect multi-cluster)
    ("Ambiguous", "Is Pluto a planet?", "No, Pluto is classified as a dwarf planet"),
    ("Ambiguous", "Is a tomato a fruit or a vegetable?", "Botanically a tomato is a fruit"),
    # Trick (expect uncertainty or hallucination)
    ("Trick", "Who was the first person to walk on Mars?", "No one has walked on Mars yet"),
    (
        "Trick",
        "What year was the city of Atlantis founded?",
        "Atlantis is a mythological city and was never actually founded",
    ),
    (
        "Trick",
        "What is the world record for running a mile in under 2 minutes?",
        "No one has run a mile in under 2 minutes",
    ),
    # Subjective (skipped for calibration, no ground truth)
    ("Subjective", "What is the best programming language?", None),
    ("Subjective", "Is a hot dog a sandwich?", None),
]


@dataclass
class CalibrationResult:
    """Results for a single question including correctness and confidence."""

    category: str
    question: str
    ground_truth: str
    responses: list[str]
    log_likelihoods: list[float]
    semantic_ids: list[int]
    entropy_discrete: float
    entropy_weighted: float
    mode_cluster_discrete: int
    confidence_discrete: float
    mode_cluster_weighted: int
    confidence_weighted: float
    is_correct_discrete: bool
    is_correct_weighted: bool
    cluster_responses: dict[int, list[str]] = field(default_factory=dict)

    @property
    def num_clusters(self) -> int:
        """Return the number of distinct semantic clusters."""
        return len(set(self.semantic_ids))


def format_result(result: CalibrationResult, index: int, total: int) -> str:
    """Format a single question's calibration results."""
    corr_d = "CORRECT" if result.is_correct_discrete else "INCORRECT"
    corr_w = "CORRECT" if result.is_correct_weighted else "INCORRECT"
    lines = [
        f"=== Question {index}/{total} [{result.category}] ===",
        f"Q: {result.question}",
        f"Ground truth: {result.ground_truth}",
        f"Mode cluster correct — discrete: {corr_d}, weighted: {corr_w}",
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
        is_mode_d = " [MODE-discrete]" if cid == result.mode_cluster_discrete else ""
        is_mode_w = " [MODE-weighted]" if cid == result.mode_cluster_weighted else ""
        lines.append(f"  [Cluster {cid}] ({len(members)} responses){is_mode_d}{is_mode_w}")
        for idx, text, ll in members:
            display = text[:180] + "..." if len(text) > 180 else text
            lines.append(f"    {idx} (ll={ll:.3f}): {display!r}")

    lines.extend(
        [
            "",
            f"Entropy  — discrete: {result.entropy_discrete:.4f}, weighted: {result.entropy_weighted:.4f}",
            f"Confidence — discrete: {result.confidence_discrete:.4f}, weighted: {result.confidence_weighted:.4f}",
        ]
    )
    return "\n".join(lines)


def print_calibration_summary(results: list[CalibrationResult]) -> None:
    """Print per-question summary and aggregate calibration metrics."""
    print("\n" + "=" * 90)
    print("Per-Question Summary")
    print("-" * 90)
    print(
        f"{'Question':<35} {'Cor(d)':>6} {'Cor(w)':>6} {'Clust':>5}"
        f" {'Conf(d)':>8} {'Conf(w)':>8} {'SE(d)':>7} {'SE(w)':>7}"
    )
    print("-" * 90)
    for r in results:
        q = r.question[:33] + ".." if len(r.question) > 35 else r.question
        cd = "Y" if r.is_correct_discrete else "N"
        cw = "Y" if r.is_correct_weighted else "N"
        print(
            f"{q:<35} {cd:>6} {cw:>6} {r.num_clusters:>5} "
            f"{r.confidence_discrete:>8.4f} {r.confidence_weighted:>8.4f} "
            f"{r.entropy_discrete:>7.4f} {r.entropy_weighted:>7.4f}"
        )

    conf_d = np.array([r.confidence_discrete for r in results])
    conf_w = np.array([r.confidence_weighted for r in results])
    corr_d = np.array([float(r.is_correct_discrete) for r in results])
    corr_w = np.array([float(r.is_correct_weighted) for r in results])

    print("\n" + "=" * 90)
    print("Calibration Metrics (pre-calibration)")
    print("-" * 90)
    print(f"  {'':20} {'Discrete':>12} {'Weighted':>12}")
    ece_d = expected_calibration_error(conf_d, corr_d)
    ece_w = expected_calibration_error(conf_w, corr_w)
    ace_d = average_calibration_error(conf_d, corr_d)
    ace_w = average_calibration_error(conf_w, corr_w)
    print(f"  {'ECE (10 bins)':<20} {ece_d:>12.4f} {ece_w:>12.4f}")
    print(f"  {'ACE (unbinned)':<20} {ace_d:>12.4f} {ace_w:>12.4f}")

    n = len(results)
    print(f"\n  Note: N={n} is very small — metrics are illustrative, not statistically robust.")

    print("\n" + "=" * 90)
    print("Post-hoc Calibration (LOOCV)")
    print("-" * 90)
    print(f"  {'Method':<20} {'ACE(d)':>10} {'ACE(w)':>10}")
    print(f"  {'(uncalibrated)':<20} {ace_d:>10.4f} {ace_w:>10.4f}")
    for name, cls in CALIBRATORS:
        ace_d, _ = leave_one_out_evaluate(conf_d, corr_d, cls)
        ace_w, _ = leave_one_out_evaluate(conf_w, corr_w, cls)
        print(f"  {name:<20} {ace_d:>10.4f} {ace_w:>10.4f}")
    print("=" * 90)


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
        default=1.0,
        help="Sampling temperature for Gemma (default: 1.0).",
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

    evaluable = [(cat, q, gt) for cat, q, gt in QUESTIONS if gt is not None]
    print(
        f"\nRunning semantic calibration: {len(evaluable)} evaluable questions, "
        f"{args.num_samples} samples each, temperature={args.temperature}\n"
    )

    results: list[CalibrationResult] = []
    for i, (category, question, ground_truth) in enumerate(evaluable, 1):
        print(f"[{i}/{len(evaluable)}] Generating responses for: {question}")
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

        mode_d, conf_d = compute_semantic_confidence_discrete(semantic_ids)
        mode_w, conf_w = compute_semantic_confidence_weighted(semantic_ids, log_likelihoods)

        # Build cluster -> responses mapping for correctness check
        cluster_resps: dict[int, list[str]] = {}
        for sid, resp in zip(semantic_ids, responses, strict=True):
            cluster_resps.setdefault(sid, []).append(resp)

        # Check correctness of mode clusters (may differ between variants)
        is_correct_d = check_cluster_correctness(cluster_resps[mode_d], ground_truth, entailment)
        is_correct_w = (
            is_correct_d
            if mode_w == mode_d
            else check_cluster_correctness(cluster_resps[mode_w], ground_truth, entailment)
        )

        result = CalibrationResult(
            category=category,
            question=question,
            ground_truth=ground_truth,
            responses=responses,
            log_likelihoods=log_likelihoods,
            semantic_ids=semantic_ids,
            entropy_discrete=se_discrete,
            entropy_weighted=se_weighted,
            mode_cluster_discrete=mode_d,
            confidence_discrete=conf_d,
            mode_cluster_weighted=mode_w,
            confidence_weighted=conf_w,
            is_correct_discrete=is_correct_d,
            is_correct_weighted=is_correct_w,
            cluster_responses=cluster_resps,
        )
        results.append(result)
        print(format_result(result, i, len(evaluable)))
        print()

    print_calibration_summary(results)


if __name__ == "__main__":
    main()
