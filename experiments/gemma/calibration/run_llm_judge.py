"""Post-hoc LLM judge evaluation on existing calibration results.

Reads a result JSON from generate.py, sends each question's mode-cluster
representative to a Claude judge, and writes an augmented JSON with both NLI
and LLM judge correctness labels plus comparison metrics.

Usage:
    uv run python gemma/calibration/run_llm_judge.py \
        --input data/results/smoke_test.json \
        --output data/results/smoke_test_judged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import anthropic
from core.calibration import (
    compute_aggregates,
    compute_semantic_confidence_discrete,
    compute_semantic_confidence_weighted,
)
from core.llm_judge import DEFAULT_JUDGE_SYSTEM_PROMPT, JudgeModel, judge_correctness
import numpy as np


def load_results(path: Path) -> dict:
    """Load a result JSON file."""
    with path.open() as f:
        return json.load(f)


def _find_representative(semantic_ids: list[int], responses: list[str], mode_id: int) -> str:
    """Return the first response belonging to the given cluster."""
    for sid, resp in zip(semantic_ids, responses, strict=True):
        if sid == mode_id:
            return resp
    return responses[0]


def judge_results(
    data: dict,
    model: JudgeModel = "claude-sonnet-4-6-20250514",
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
) -> dict:
    """Run LLM judge on all results and return augmented data."""
    results = data["results"]
    total = len(results)
    client = anthropic.Anthropic()

    print(f"\nJudging {total} questions with {model}...\n")

    for i, result in enumerate(results):
        question = result["question"]
        answer_aliases = result["answer_aliases"]
        semantic_ids = result["semantic_ids"]
        responses = result["responses"]
        t0 = time.time()

        print(f"[{i + 1}/{total}] {question[:80]}")

        mode_d, _ = compute_semantic_confidence_discrete(semantic_ids)
        mode_w, _ = compute_semantic_confidence_weighted(
            semantic_ids,
            result["log_likelihoods"],
        )

        rep_d = _find_representative(semantic_ids, responses, mode_d)
        is_correct_d = judge_correctness(
            question,
            rep_d,
            answer_aliases,
            model,
            system_prompt,
            client,
        )

        if mode_w == mode_d:
            is_correct_w = is_correct_d
        else:
            rep_w = _find_representative(semantic_ids, responses, mode_w)
            is_correct_w = judge_correctness(
                question,
                rep_w,
                answer_aliases,
                model,
                system_prompt,
                client,
            )

        result["is_correct_discrete_llm"] = is_correct_d
        result["is_correct_weighted_llm"] = is_correct_w
        result["llm_judge_model"] = model

        elapsed = time.time() - t0
        nli_agree = result["is_correct_discrete"] == is_correct_d
        correct = "Y" if is_correct_d else "N"
        agree = "agree" if nli_agree else "DISAGREE"
        print(f"  -> correct={correct}, NLI {agree} ({elapsed:.1f}s)")

    llm_agg = compute_aggregates(
        results,
        correctness_key_discrete="is_correct_discrete_llm",
        correctness_key_weighted="is_correct_weighted_llm",
    )

    nli_d = np.array([r["is_correct_discrete"] for r in results])
    llm_d = np.array([r["is_correct_discrete_llm"] for r in results])
    agreement = float(np.mean(nli_d == llm_d))

    augmented = dict(data)
    augmented["aggregate"] = {
        **data["aggregate"],
        "ece_discrete_llm": llm_agg["ece_discrete"],
        "ece_weighted_llm": llm_agg["ece_weighted"],
        "ace_discrete_llm": llm_agg["ace_discrete"],
        "ace_weighted_llm": llm_agg["ace_weighted"],
        "accuracy_llm": llm_agg["accuracy"],
        "judge_nli_agreement": agreement,
    }
    augmented["metadata"] = {
        **data["metadata"],
        "llm_judge_model": model,
    }

    return augmented


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON result file from generate.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for augmented JSON with LLM judge results.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_JUDGE_SYSTEM_PROMPT,
        help="Custom system prompt for the judge.",
    )
    return parser.parse_args()


def main() -> None:
    """Run post-hoc LLM judge evaluation."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading results from {input_path}")
    data = load_results(input_path)
    n = len(data["results"])
    print(f"  {n} questions, T={data['metadata']['temperature']}")

    augmented = judge_results(data, system_prompt=args.system_prompt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(augmented, f, indent=2)
    print(f"\nAugmented results saved to {output_path}")

    agg = augmented["aggregate"]
    print("\n--- NLI vs LLM Judge Comparison ---")
    print(f"Agreement:     {agg['judge_nli_agreement']:.1%}")
    print(f"Accuracy NLI:  {agg['accuracy']:.1%}")
    print(f"Accuracy LLM:  {agg['accuracy_llm']:.1%}")
    print(f"ACE NLI (d):   {agg['ace_discrete']:.4f}")
    print(f"ACE LLM (d):   {agg['ace_discrete_llm']:.4f}")


if __name__ == "__main__":
    main()
