"""Semantic calibration experiment on TriviaQA with Gemma.

Generates multiple responses per question, clusters semantically, checks
correctness against ground-truth answer aliases, and saves results as JSON.
Supports incremental saving (JSONL partial file) and resume after crash.

Usage:
    uv run python gemma/calibration/run_experiment.py \
        --num-questions 200 --num-samples 10 --temperature 0.7 \
        --seed 42 --output data/results/run_t07_s42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from core import (
    CACHE_DIR,
    DATA_DIR,
    EntailmentModel,
    cluster_assignment_entropy,
    generate_responses,
    get_semantic_ids,
    suppress_hf_noise,
    weighted_semantic_entropy,
)
from core.calibration import (
    average_calibration_error,
    compute_semantic_confidence_discrete,
    compute_semantic_confidence_weighted,
    expected_calibration_error,
)
from core.correctness import check_cluster_correctness
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"


def load_trivia_questions(
    num_questions: int,
    seed: int,
) -> list[dict]:
    """Load and sample questions from TriviaQA validation split.

    Returns:
        List of dicts with keys: ``question``, ``answer_aliases``.
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(indices)
    selected = [ds[i] for i in indices[:num_questions]]

    questions = []
    for item in selected:
        aliases = item["answer"]["aliases"]
        if item["answer"]["value"] not in aliases:
            aliases = [item["answer"]["value"], *aliases]
        questions.append(
            {
                "question": item["question"],
                "answer_aliases": aliases,
            }
        )
    return questions


def check_correctness_multi_alias(
    cluster_responses: list[str],
    answer_aliases: list[str],
    entailment_model: EntailmentModel,
) -> bool:
    """Check if a cluster is correct against any answer alias."""
    return any(check_cluster_correctness(cluster_responses, alias, entailment_model) for alias in answer_aliases)


def load_partial_results(partial_path: Path) -> list[dict]:
    """Load previously completed results from JSONL partial file."""
    if not partial_path.exists():
        return []
    results = []
    with partial_path.open() as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                results.append(json.loads(stripped))
    return results


def append_partial_result(partial_path: Path, result: dict) -> None:
    """Append a single result to the JSONL partial file."""
    with partial_path.open("a") as f:
        f.write(json.dumps(result) + "\n")


def process_question(
    question_data: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    entailment: EntailmentModel,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    """Run the full pipeline for a single question."""
    question = question_data["question"]
    answer_aliases = question_data["answer_aliases"]

    responses, log_likelihoods = generate_responses(
        question,
        model,
        tokenizer,
        num_samples,
        temperature,
        max_new_tokens,
    )

    semantic_ids = get_semantic_ids(responses, entailment)
    se_discrete = cluster_assignment_entropy(semantic_ids)
    se_weighted = weighted_semantic_entropy(semantic_ids, log_likelihoods)

    mode_d, conf_d = compute_semantic_confidence_discrete(semantic_ids)
    mode_w, conf_w = compute_semantic_confidence_weighted(
        semantic_ids,
        log_likelihoods,
    )

    cluster_resps: dict[int, list[str]] = {}
    for sid, resp in zip(semantic_ids, responses, strict=True):
        cluster_resps.setdefault(sid, []).append(resp)

    is_correct_d = check_correctness_multi_alias(
        cluster_resps[mode_d],
        answer_aliases,
        entailment,
    )
    is_correct_w = (
        is_correct_d
        if mode_w == mode_d
        else check_correctness_multi_alias(
            cluster_resps[mode_w],
            answer_aliases,
            entailment,
        )
    )

    return {
        "question": question,
        "answer_aliases": answer_aliases,
        "responses": responses,
        "log_likelihoods": log_likelihoods,
        "semantic_ids": semantic_ids,
        "confidence_discrete": conf_d,
        "confidence_weighted": conf_w,
        "is_correct_discrete": is_correct_d,
        "is_correct_weighted": is_correct_w,
        "entropy_discrete": se_discrete,
        "entropy_weighted": se_weighted,
        "num_clusters": len(set(semantic_ids)),
    }


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate calibration metrics from per-question results."""
    conf_d = np.array([r["confidence_discrete"] for r in results])
    conf_w = np.array([r["confidence_weighted"] for r in results])
    corr_d = np.array([float(r["is_correct_discrete"]) for r in results])
    corr_w = np.array([float(r["is_correct_weighted"]) for r in results])

    return {
        "ece_discrete": float(expected_calibration_error(conf_d, corr_d)),
        "ece_weighted": float(expected_calibration_error(conf_w, corr_w)),
        "ace_discrete": float(average_calibration_error(conf_d, corr_d)),
        "ace_weighted": float(average_calibration_error(conf_w, corr_w)),
        "accuracy": float(corr_d.mean()),
    }


def save_final_results(
    output_path: Path,
    metadata: dict,
    results: list[dict],
) -> None:
    """Write the consolidated JSON result file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": results,
        "aggregate": compute_aggregates(results),
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-questions",
        type=int,
        default=200,
        help="Number of TriviaQA questions (default: 200).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Responses per question (default: 10).",
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
        help="Max tokens per response (default: 128).",
    )
    parser.add_argument(
        "--nli-model",
        default="microsoft/deberta-base-mnli",
        help="NLI model for entailment checking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "results" / "run.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the semantic calibration experiment."""
    args = parse_args()
    torch.manual_seed(args.seed)

    suppress_hf_noise()

    output_path = Path(args.output)
    partial_path = Path(str(output_path) + ".partial")

    # Resume support: load any previously completed results
    completed = load_partial_results(partial_path)
    start_idx = len(completed)

    print(f"Loading TriviaQA (sampling {args.num_questions} questions, seed={args.seed})...")
    questions = load_trivia_questions(args.num_questions, args.seed)

    if start_idx > 0:
        print(f"Resuming from question {start_idx + 1} ({start_idx} already completed)")

    print(f"Loading Gemma model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
    model.eval()

    print(f"Loading NLI model: {args.nli_model}")
    entailment = EntailmentModel(args.nli_model)

    total = len(questions)
    print(f"\nRunning: {total} questions, {args.num_samples} samples each, temperature={args.temperature}\n")

    results = list(completed)
    for i in range(start_idx, total):
        t0 = time.time()
        q = questions[i]
        print(f"[{i + 1}/{total}] {q['question'][:80]}")

        result = process_question(
            q,
            model,
            tokenizer,
            entailment,
            args.num_samples,
            args.temperature,
            args.max_new_tokens,
        )
        results.append(result)
        append_partial_result(partial_path, result)

        elapsed = time.time() - t0
        correct = "Y" if result["is_correct_discrete"] else "N"
        print(
            f"  -> clusters={result['num_clusters']}, "
            f"conf={result['confidence_discrete']:.3f}, "
            f"correct={correct}, "
            f"entropy={result['entropy_discrete']:.3f} "
            f"({elapsed:.1f}s)"
        )

    # Consolidate into final JSON
    metadata = {
        "model": MODEL_ID,
        "nli_model": args.nli_model,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "num_questions": total,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_final_results(output_path, metadata, results)
    print(f"\nResults saved to {output_path}")

    # Print summary
    agg = compute_aggregates(results)
    print(f"Accuracy: {agg['accuracy']:.1%}")
    print(f"ECE (discrete): {agg['ece_discrete']:.4f}")
    print(f"ACE (discrete): {agg['ace_discrete']:.4f}")

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()
        print(f"Cleaned up {partial_path}")


if __name__ == "__main__":
    main()
