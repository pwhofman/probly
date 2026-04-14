"""Step 1: Generate responses, cluster semantically, compute entropy/confidence.

Generates multiple responses per question from Gemma, clusters them via
NLI-based semantic equivalence, and computes entropy and confidence scores.
Does NOT perform correctness checking -- that is deferred to ``analyze.py``.

Supports incremental saving (JSONL partial file) and resume after crash.

Usage:
    uv run python gemma/calibration/generate.py \
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
    DEFAULT_NLI_MODEL,
    EntailmentModel,
    NLIModel,
    cluster_assignment_entropy,
    generate_responses,
    get_semantic_ids,
    suppress_hf_noise,
    weighted_semantic_entropy,
)
from core.calibration import (
    compute_semantic_confidence_discrete,
    compute_semantic_confidence_weighted,
)
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma import MODEL_ID


def load_trivia_questions(
    num_questions: int,
    seed: int,
) -> list[dict]:
    """Load and sample questions from TriviaQA validation split.

    Returns:
        List of dicts with keys: ``question``, ``answer_aliases``.
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
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
    partial_path.parent.mkdir(parents=True, exist_ok=True)
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
    """Run generation and semantic clustering for a single question."""
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

    # Build clusters dict keyed by semantic ID
    clusters: dict[str, dict] = {}
    for sid, resp, idx in zip(semantic_ids, responses, range(len(responses)), strict=False):
        key = str(sid)
        if key not in clusters:
            clusters[key] = {"representative": resp, "count": 0, "response_indices": []}
        clusters[key]["count"] += 1
        clusters[key]["response_indices"].append(idx)

    return {
        "question": question,
        "answer_aliases": answer_aliases,
        "responses": responses,
        "log_likelihoods": log_likelihoods,
        "semantic_ids": semantic_ids,
        "clusters": clusters,
        "mode_cluster_discrete": mode_d,
        "mode_cluster_weighted": mode_w,
        "confidence_discrete": conf_d,
        "confidence_weighted": conf_w,
        "is_correct_discrete": None,
        "is_correct_weighted": None,
        "entropy_discrete": se_discrete,
        "entropy_weighted": se_weighted,
        "num_clusters": len(set(semantic_ids)),
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
        "aggregate": None,
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
        default=DEFAULT_NLI_MODEL,
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


def generate_main(
    *,
    num_questions: int = 200,
    num_samples: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    nli_model: NLIModel = DEFAULT_NLI_MODEL,
    seed: int = 42,
    output: str | Path | None = None,
) -> Path:
    """Run the generation and semantic clustering pipeline.

    Args:
        num_questions: Number of TriviaQA questions to sample.
        num_samples: Number of responses to generate per question.
        temperature: Sampling temperature for generation.
        max_new_tokens: Maximum tokens per generated response.
        nli_model: HuggingFace model ID for NLI-based entailment.
        seed: Random seed for reproducibility.
        output: Path for the output JSON file.

    Returns:
        Path to the saved results JSON file.
    """
    torch.manual_seed(seed)
    suppress_hf_noise()

    output_path = Path(output) if output is not None else DATA_DIR / "results" / "run.json"
    partial_path = Path(str(output_path) + ".partial")

    # Resume support: load any previously completed results
    completed = load_partial_results(partial_path)
    start_idx = len(completed)

    print(f"Loading TriviaQA (sampling {num_questions} questions, seed={seed})...")
    questions = load_trivia_questions(num_questions, seed)

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

    print(f"Loading NLI model: {nli_model}")
    entailment = EntailmentModel(nli_model)

    total = len(questions)
    print(f"\nRunning: {total} questions, {num_samples} samples each, temperature={temperature}\n")

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
            num_samples,
            temperature,
            max_new_tokens,
        )
        results.append(result)
        append_partial_result(partial_path, result)

        elapsed = time.time() - t0
        print(
            f"  -> clusters={result['num_clusters']}, "
            f"conf={result['confidence_discrete']:.3f}, "
            f"entropy={result['entropy_discrete']:.3f} "
            f"({elapsed:.1f}s)"
        )

    # Consolidate into final JSON
    metadata = {
        "step": "generate",
        "model": MODEL_ID,
        "nli_model": nli_model,
        "temperature": temperature,
        "num_samples": num_samples,
        "num_questions": total,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_final_results(output_path, metadata, results)
    print(f"\nResults saved to {output_path}")

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()
        print(f"Cleaned up {partial_path}")

    return output_path


if __name__ == "__main__":
    args = parse_args()
    generate_main(
        num_questions=args.num_questions,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        nli_model=args.nli_model,
        seed=args.seed,
        output=args.output,
    )
