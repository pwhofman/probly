"""Benchmark batched NLI querying for semantic entropy clustering."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from probly.representation.text_generation.torch import TorchTextGeneration
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer


NLI_MODEL_NAME = "microsoft/deberta-base-mnli"
SAMPLES = [
    "Semantic entropy measures uncertainty by grouping generated answers by meaning.",
    "Predictive entropy can be high when a model spreads probability across many token sequences.",
    "Two answers may use different words while still making the same factual claim.",
    "A contradiction between responses indicates that they should not share a semantic cluster.",
    "Natural language inference models can compare whether one statement entails another.",
    "Batching NLI pairs reduces the number of separate model calls needed for clustering.",
    "The capital of France is Paris.",
    "Water is commonly represented by the chemical formula H2O.",
    "The Moon orbits Earth approximately once every month.",
    "A hot dog classification question can have multiple defensible interpretations.",
    "Semantic entropy measures uncertainty by grouping generated answers by meaning.",
    "Predictive entropy can be high when a model spreads probability across many token sequences.",
    "Two answers may use different words while still making the same factual claim.",
    "A contradiction between responses indicates that they should not share a semantic cluster.",
    "Natural language inference models can compare whether one statement entails another.",
    "Batching NLI pairs reduces the number of separate model calls needed for clustering.",
    "The capital of France is Paris.",
    "Water is commonly represented by the chemical formula H2O.",
    "The Moon orbits Earth approximately once every month.",
    "A hot dog classification question can have multiple defensible interpretations.",
]
SAMPLE_COUNTS = range(2,21)
REPETITIONS = 5
UNBATCHED_BATCH_SIZE = 1
BATCHED_BATCH_SIZE = 1000
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "semantic_entropy_nli_batching.json"


class CountingHFGreedySemanticClusterer(HFGreedySemanticClusterer):
    """Greedy semantic clusterer that records NLI pair query counts."""

    nli_pairs_scored: int
    nli_model_batches: int

    def reset_counts(self) -> None:
        """Reset query counters before one benchmark run."""
        self.nli_pairs_scored = 0
        self.nli_model_batches = 0

    def _predict_pair_labels(self, statements: np.ndarray, pairs: torch.Tensor) -> torch.Tensor:
        self.nli_pairs_scored += int(pairs.shape[0])
        self.nli_model_batches += math.ceil(pairs.shape[0] / self.batch_size)
        return super()._predict_pair_labels(statements, pairs)


def synchronize_device() -> None:
    """Synchronize CUDA timing when running on a GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_generation(samples: list[str]) -> TorchTextGeneration:
    """Create a text generation object for semantic clustering.

    Args:
        samples: Text samples to cluster.

    Returns:
        Text generation with one batch row and samples along axis 1.
    """
    return TorchTextGeneration(
        text=np.array([samples], dtype=object),
        log_likelihood=torch.zeros((1, len(samples)), dtype=torch.float32),
    )


def runtime_seconds(
    clusterer: CountingHFGreedySemanticClusterer,
    samples: list[str],
) -> tuple[float, int, int]:
    """Measure one semantic clustering invocation.

    Args:
        clusterer: Configured semantic clusterer.
        samples: Text samples to cluster.

    Returns:
        Runtime, number of NLI pairs scored, and number of NLI model batches.
    """
    clusterer.reset_counts()
    generation = make_generation(samples)
    synchronize_device()
    start = perf_counter()
    clusterer(generation, axis=1)
    synchronize_device()
    return perf_counter() - start, clusterer.nli_pairs_scored, clusterer.nli_model_batches


def benchmark_sample_count(
    unbatched: CountingHFGreedySemanticClusterer,
    batched: CountingHFGreedySemanticClusterer,
    num_samples: int,
) -> dict[str, object]:
    """Benchmark unbatched and batched NLI querying for one sample count.

    Args:
        unbatched: Clusterer with ``batch_size=1``.
        batched: Clusterer with larger NLI batch size.
        num_samples: Number of fixed samples to cluster.

    Returns:
        Raw timings and median summary for the sample count.
    """
    samples = SAMPLES[:num_samples]
    unbatched_runs = []
    batched_runs = []

    for _ in range(REPETITIONS):
        seconds, num_pairs, num_batches = runtime_seconds(unbatched, samples)
        unbatched_runs.append({"seconds": seconds, "num_pairs": num_pairs, "num_model_batches": num_batches})

    for _ in range(REPETITIONS):
        seconds, num_pairs, num_batches = runtime_seconds(batched, samples)
        batched_runs.append({"seconds": seconds, "num_pairs": num_pairs, "num_model_batches": num_batches})

    unbatched_seconds = median(run["seconds"] for run in unbatched_runs)
    batched_seconds = median(run["seconds"] for run in batched_runs)

    return {
        "num_samples": num_samples,
        "unbatched_batch_size": UNBATCHED_BATCH_SIZE,
        "batched_batch_size": BATCHED_BATCH_SIZE,
        "unbatched_runs": unbatched_runs,
        "batched_runs": batched_runs,
        "num_pairs": max(run["num_pairs"] for run in [*unbatched_runs, *batched_runs]),
        "unbatched_seconds": unbatched_seconds,
        "batched_seconds": batched_seconds,
        "speedup": unbatched_seconds / batched_seconds if batched_seconds > 0 else None,
    }


def main() -> None:
    """Run the NLI batching benchmark and write raw results as JSON."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    print(f"Loading NLI model: {NLI_MODEL_NAME}")
    unbatched = CountingHFGreedySemanticClusterer.from_model_name(
        NLI_MODEL_NAME,
        model_kwargs=model_kwargs,
        batch_size=UNBATCHED_BATCH_SIZE,
    )
    batched = CountingHFGreedySemanticClusterer(
        unbatched.model,
        unbatched.tokenizer,
        batch_size=BATCHED_BATCH_SIZE,
    )

    print("Running warmup NLI clustering")
    runtime_seconds(unbatched, SAMPLES[:2])

    rows = []
    for num_samples in SAMPLE_COUNTS:
        print(f"Benchmarking NLI clustering with {num_samples} sample(s)")
        rows.append(benchmark_sample_count(unbatched, batched, num_samples))

    output = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "nli_model": NLI_MODEL_NAME,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "samples": SAMPLES,
            "sample_counts": list(SAMPLE_COUNTS),
            "repetitions": REPETITIONS,
            "unbatched_batch_size": UNBATCHED_BATCH_SIZE,
            "batched_batch_size": BATCHED_BATCH_SIZE,
        },
        "results": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote NLI benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
