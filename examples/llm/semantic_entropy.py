"""Compute semantic entropy by clustering sampled LLM answers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

from probly.quantification.measure.distribution import entropy
from probly.representer.sampler.huggingface import HFTextGenerationSampler
from probly.representer.semantic_clustering import GreedyHFSemanticClusterer


# GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
NLI_MODEL_NAME = "microsoft/deberta-base-mnli"

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


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Semantic entropy results for a single question."""

    category: str
    question: str
    responses: list[str]
    log_likelihoods: list[float]
    group_ids: list[int]
    entropy_discrete: float
    entropy_weighted: float

    @property
    def num_clusters(self) -> int:
        """Return the number of distinct semantic clusters."""
        return len(set(self.group_ids))


def format_result(result: QuestionResult, index: int, total: int) -> str:
    """Format one question's semantic entropy result for console output."""
    lines = [
        f"=== Question {index}/{total} [{result.category}] ===",
        f"Q: {result.question}",
        "",
        f"Responses ({len(result.responses)} samples, {result.num_clusters} clusters):",
    ]

    clusters: dict[int, list[tuple[int, str, float]]] = {}
    for sample_index, (group_id, response, log_likelihood) in enumerate(
        zip(result.group_ids, result.responses, result.log_likelihoods, strict=True),
        start=1,
    ):
        clusters.setdefault(group_id, []).append((sample_index, response, log_likelihood))

    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        lines.append(f"  [Cluster {cluster_id}] ({len(members)} responses)")
        for sample_index, response, log_likelihood in members:
            display = response[:180] + "..." if len(response) > 180 else response
            lines.append(f"    {sample_index} (ll={log_likelihood:.3f}): {display!r}")

    lines.extend(
        [
            "",
            f"Semantic Entropy (discrete): {result.entropy_discrete:.4f}",
            f"Semantic Entropy (weighted): {result.entropy_weighted:.4f}",
        ]
    )
    return "\n".join(lines)


def print_summary(results: list[QuestionResult]) -> None:
    """Print a compact summary table for all questions."""
    print("\n" + "=" * 78)
    print("Summary")
    print("-" * 78)
    print(f"{'Question':<40} {'Clusters':>8} {'Discrete':>10} {'Weighted':>10}")
    print("-" * 78)
    for result in results:
        question = result.question[:38] + ".." if len(result.question) > 40 else result.question
        print(
            f"{question:<40} {result.num_clusters:>8} "
            f"{result.entropy_discrete:>10.4f} {result.entropy_weighted:>10.4f}"
        )
    print("=" * 78)


def main() -> None:
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    generation_model_kwargs: dict[str, object] = {"dtype": "auto"}
    nli_model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        generation_model_kwargs["device_map"] = "auto"
        nli_model_kwargs["device_map"] = "auto"

    print(f"Loading generation model: {GENERATION_MODEL_NAME}")
    sampler = HFTextGenerationSampler.from_model_name(
        GENERATION_MODEL_NAME,
        num_samples=10,
        model_kwargs=generation_model_kwargs,
        batch_size=4,
        temperature=0.7,
        max_new_tokens=128,
    )
    print(f"Loading NLI model: {NLI_MODEL_NAME}")
    clusterer = GreedyHFSemanticClusterer.from_model_name(
        NLI_MODEL_NAME,
        model_kwargs=nli_model_kwargs,
        batch_size=10,
    )

    questions = [question for _, question in QUESTIONS]
    chats = [[{"role": "user", "content": question}] for question in questions]

    print(
        f"\nRunning semantic entropy: {len(QUESTIONS)} questions, "
        f"{sampler.num_samples} samples each, temperature={sampler.temperature}\n"
    )
    print("Generating responses...")
    text_sample = sampler(chats)
    print("Clustering responses...")
    semantic_sample = clusterer(text_sample)
    discrete_entropies = entropy(semantic_sample.uniform_logits())
    weighted_entropies = entropy(semantic_sample)

    results: list[QuestionResult] = []
    for index, ((category, question), responses, log_likelihoods, group_ids, discrete_entropy, weighted_entropy) in enumerate(
        zip(
            QUESTIONS,
            text_sample.tensor.text,
            text_sample.tensor.log_likelihood,
            semantic_sample.group_ids,
            discrete_entropies,
            weighted_entropies,
            strict=True,
        ),
        start=1,
    ):
        response_list = [str(response).strip() for response in responses]
        log_likelihood_list = [float(log_likelihood.item()) for log_likelihood in log_likelihoods]
        group_id_list = [int(group_id.item()) for group_id in group_ids]
        result = QuestionResult(
            category=category,
            question=question,
            responses=response_list,
            log_likelihoods=log_likelihood_list,
            group_ids=group_id_list,
            entropy_discrete=float(discrete_entropy.item()),
            entropy_weighted=float(weighted_entropy.item()),
        )
        results.append(result)
        print(format_result(result, index, len(QUESTIONS)))
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
