"""Benchmark batched response sampling for semantic entropy examples."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import torch

from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model


GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
QUESTIONS = [
    "What is a quick way to explain semantic entropy for language models?",
    "Why can two plausible answers to the same question express different meanings?",
]
SAMPLE_COUNTS = range(1, 21)
REPETITIONS = 3
MAX_NEW_TOKENS = 64
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "semantic_entropy_generation_batching.json"


def synchronize_device() -> None:
    """Synchronize CUDA timing when running on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def runtime_seconds(sampler: HFTextGenerationSampler) -> float:
    """Measure one sampler invocation in seconds.

    Args:
        sampler: Configured Hugging Face text generation sampler.

    Returns:
        Wall-clock runtime for generating samples for ``QUESTIONS``.
    """
    synchronize_device()
    start = perf_counter()
    sampler(QUESTIONS)
    synchronize_device()
    return perf_counter() - start


def make_sampler(model: Any, tokenizer: Any, *, num_samples: int, batch_size: int) -> HFTextGenerationSampler:
    """Create a sampler with benchmark settings.

    Args:
        model: Loaded generation model.
        tokenizer: Tokenizer associated with ``model``.
        num_samples: Number of responses to sample per question.
        batch_size: Number of samples to generate per model call.

    Returns:
        Configured text generation sampler.
    """
    return HFTextGenerationSampler(
        model,
        tokenizer,
        num_samples=num_samples,
        batch_size=batch_size,
        temperature=0.7,
        max_new_tokens=MAX_NEW_TOKENS,
        top_k=50,
    )


def benchmark_sample_count(model: Any, tokenizer: Any, num_samples: int) -> dict[str, object]:
    """Benchmark unbatched and fully batched sampling for one sample count.

    Args:
        model: Loaded generation model.
        tokenizer: Tokenizer associated with ``model``.
        num_samples: Number of responses to sample per question.

    Returns:
        Raw timings and median summary for the sample count.
    """
    unbatched = make_sampler(model, tokenizer, num_samples=num_samples, batch_size=1)
    batched = make_sampler(model, tokenizer, num_samples=num_samples, batch_size=num_samples)

    unbatched_runs = [runtime_seconds(unbatched) for _ in range(REPETITIONS)]
    batched_runs = [runtime_seconds(batched) for _ in range(REPETITIONS)]
    unbatched_seconds = median(unbatched_runs)
    batched_seconds = median(batched_runs)

    return {
        "num_samples": num_samples,
        "num_questions": len(QUESTIONS),
        "unbatched_batch_size": 1,
        "batched_batch_size": num_samples,
        "unbatched_runs_seconds": unbatched_runs,
        "batched_runs_seconds": batched_runs,
        "unbatched_seconds": unbatched_seconds,
        "batched_seconds": batched_seconds,
        "speedup": unbatched_seconds / batched_seconds if batched_seconds > 0 else None,
    }


def main() -> None:
    """Run the generation batching benchmark and write raw results as JSON."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    model_kwargs: dict[str, object] = {"dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    print(f"Loading generation model: {GENERATION_MODEL_NAME}")
    model, tokenizer = load_model(GENERATION_MODEL_NAME, model_kwargs=model_kwargs)
    model.eval()

    print("Running warmup generation")
    runtime_seconds(make_sampler(model, tokenizer, num_samples=1, batch_size=1))

    rows = []
    for num_samples in SAMPLE_COUNTS:
        print(f"Benchmarking generation with {num_samples} sample(s)")
        rows.append(benchmark_sample_count(model, tokenizer, num_samples))

    output = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "generation_model": GENERATION_MODEL_NAME,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "questions": QUESTIONS,
            "sample_counts": list(SAMPLE_COUNTS),
            "repetitions": REPETITIONS,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "results": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote generation benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
