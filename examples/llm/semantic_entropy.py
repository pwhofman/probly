"""Compute semantic entropy by clustering sampled LLM answers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

from probly.quantification import decompose
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.distribution.torch_sparse_log_categorical import (
    TorchSparseLogCategoricalDistribution,
    TorchSparseLogCategoricalDistributionSample,
)
from probly.representation.text_generation.torch import TorchTextGeneration
from probly.representer.clarifier.huggingface import HFQuestionClarifier
from probly.representer.sampler.huggingface import HFTextGenerationSampler
from probly.representer.sampler.huggingface import load_model
from probly.representer.semantic_clustering.huggingface import HFGreedySemanticClusterer


# GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
NLI_MODEL_NAME = "microsoft/deberta-base-mnli"
NUM_CLARIFICATIONS = 2
NUM_ANSWERS = 10

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
    clarifications: list[str]
    responses: list[list[str]]
    log_likelihoods: list[list[float]]
    group_ids: list[list[int]]
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float

    @property
    def num_clusters(self) -> int:
        """Return the number of distinct semantic clusters."""
        return len({group_id for clarification_group_ids in self.group_ids for group_id in clarification_group_ids})


def cluster_answers_by_question(
    clusterer: HFGreedySemanticClusterer,
    answers: object,
) -> TorchSparseLogCategoricalDistributionSample:
    """Cluster all answers for each original question and return clarification-level distributions.

    Args:
        clusterer: Semantic clusterer used to assign answer samples to semantic classes.
        answers: Nested answer samples with shape ``(question, clarification, answer)``.

    Returns:
        Sparse categorical distributions sampled over the clarification axis. The semantic class ids are shared
        across clarifications of the same original question, so entropy decomposition compares compatible classes.
    """
    if not hasattr(answers, "tensor") or not hasattr(answers.tensor, "tensor"):
        msg = "answers must be a nested text generation sample."
        raise TypeError(msg)

    answer_generation = answers.tensor.tensor
    if not isinstance(answer_generation, TorchTextGeneration):
        msg = "answers must contain TorchTextGeneration values."
        raise TypeError(msg)
    if answer_generation.text.ndim != 3:
        msg = "answers must have shape (question, clarification, answer)."
        raise ValueError(msg)

    num_questions, num_clarifications, num_answers = answer_generation.text.shape
    flat_shape = (num_questions, num_clarifications * num_answers)
    flat_generation = TorchTextGeneration(
        text=answer_generation.text.reshape(flat_shape),
        log_likelihood=answer_generation.log_likelihood.reshape(flat_shape),
    )
    flat_semantic = clusterer(flat_generation, axis=1)
    if not isinstance(flat_semantic, TorchSparseLogCategoricalDistribution):
        msg = "clustering flat answers must return a sparse categorical distribution."
        raise TypeError(msg)

    semantic = TorchSparseLogCategoricalDistribution(
        group_ids=flat_semantic.group_ids.reshape((num_questions, num_clarifications, num_answers)),
        logits=flat_semantic.logits.reshape((num_questions, num_clarifications, num_answers)),
    )
    return TorchSparseLogCategoricalDistributionSample(tensor=semantic, sample_dim=1)


def format_result(result: QuestionResult, index: int, total: int) -> str:
    """Format one question's semantic entropy result for console output."""
    lines = [
        f"=== Question {index}/{total} [{result.category}] ===",
        f"Q: {result.question}",
        "",
        f"Clarifications ({len(result.clarifications)} samples):",
    ]

    for clarification_index, (clarification, responses, log_likelihoods, group_ids) in enumerate(
        zip(result.clarifications, result.responses, result.log_likelihoods, result.group_ids, strict=True),
        start=1,
    ):
        lines.append(f"  Clarification {clarification_index}: {clarification!r}")
        clusters: dict[int, list[tuple[int, str, float]]] = {}
        for sample_index, (group_id, response, log_likelihood) in enumerate(
            zip(group_ids, responses, log_likelihoods, strict=True),
            start=1,
        ):
            clusters.setdefault(group_id, []).append((sample_index, response, log_likelihood))

        for cluster_id in sorted(clusters):
            members = clusters[cluster_id]
            lines.append(f"    [Cluster {cluster_id}] ({len(members)} responses)")
            for sample_index, response, log_likelihood in members:
                display = response[:180] + "..." if len(response) > 180 else response
                lines.append(f"      {sample_index} (ll={log_likelihood:.3f}): {display!r}")

    lines.extend(
        [
            "",
            f"Total uncertainty (TU): {result.total_uncertainty:.4f}",
            f"Aleatoric uncertainty (AU): {result.aleatoric_uncertainty:.4f}",
            f"Epistemic/input uncertainty (EU): {result.epistemic_uncertainty:.4f}",
        ]
    )
    return "\n".join(lines)


def print_summary(results: list[QuestionResult]) -> None:
    """Print a compact summary table for all questions."""
    print("\n" + "=" * 82)
    print("Summary")
    print("-" * 82)
    print(f"{'Question':<40} {'Clusters':>8} {'TU':>10} {'AU':>10} {'EU':>10}")
    print("-" * 82)
    for result in results:
        question = result.question[:38] + ".." if len(result.question) > 40 else result.question
        print(
            f"{question:<40} {result.num_clusters:>8} "
            f"{result.total_uncertainty:>10.4f} {result.aleatoric_uncertainty:>10.4f} "
            f"{result.epistemic_uncertainty:>10.4f}"
        )
    print("=" * 82)


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
    generation_model, tokenizer = load_model(GENERATION_MODEL_NAME, model_kwargs=generation_model_kwargs)
    generation_model.eval()
    clarifier = HFQuestionClarifier(
        generation_model,
        tokenizer,
        num_samples=NUM_CLARIFICATIONS,
        batch_size=4,
        temperature=1.0,
        max_new_tokens=48,
    )
    sampler = HFTextGenerationSampler(
        generation_model,
        tokenizer,
        num_samples=NUM_ANSWERS,
        batch_size=4,
        temperature=0.7,
        max_new_tokens=128,
    )
    print(f"Loading NLI model: {NLI_MODEL_NAME}")
    clusterer = HFGreedySemanticClusterer.from_model_name(
        NLI_MODEL_NAME,
        model_kwargs=nli_model_kwargs,
        batch_size=10,
    )

    questions = [question for _, question in QUESTIONS]
    print(
        f"\nRunning semantic entropy with input clarification: {len(QUESTIONS)} questions, "
        f"{clarifier.num_samples} clarifications each, {sampler.num_samples} answers per clarification\n"
    )
    print("Generating clarifications...")
    clarifications = clarifier(questions)
    print("Generating responses...")
    text_sample = sampler(clarifications)
    print("Clustering responses...")
    semantic_sample = cluster_answers_by_question(clusterer, text_sample)
    dense_semantic_sample = TorchCategoricalDistributionSample(
        tensor=semantic_sample.tensor.to_categorical_distribution(),
        sample_dim=semantic_sample.sample_dim,
    )
    decomposition = decompose(dense_semantic_sample)

    results: list[QuestionResult] = []
    for index, (
        (category, question),
        question_clarifications,
        responses,
        log_likelihoods,
        group_ids,
        total_uncertainty,
        aleatoric_uncertainty,
        epistemic_uncertainty,
    ) in enumerate(
        zip(
            QUESTIONS,
            clarifications.tensor.text,
            text_sample.tensor.tensor.text,
            text_sample.tensor.tensor.log_likelihood,
            semantic_sample.tensor.group_ids,
            decomposition.total,
            decomposition.aleatoric,
            decomposition.epistemic,
            strict=True,
        ),
        start=1,
    ):
        clarification_list = [str(clarification).strip() for clarification in question_clarifications]
        response_list = [
            [str(response).strip() for response in clarification_responses]
            for clarification_responses in responses
        ]
        log_likelihood_list = [
            [float(log_likelihood.item()) for log_likelihood in clarification_log_likelihoods]
            for clarification_log_likelihoods in log_likelihoods
        ]
        group_id_list = [
            [int(group_id.item()) for group_id in clarification_group_ids]
            for clarification_group_ids in group_ids
        ]
        result = QuestionResult(
            category=category,
            question=question,
            clarifications=clarification_list,
            responses=response_list,
            log_likelihoods=log_likelihood_list,
            group_ids=group_id_list,
            total_uncertainty=float(total_uncertainty.item()),
            aleatoric_uncertainty=float(aleatoric_uncertainty.item()),
            epistemic_uncertainty=float(epistemic_uncertainty.item()),
        )
        results.append(result)
        print(format_result(result, index, len(QUESTIONS)))
        print()

    print_summary(results)


if __name__ == "__main__":
    main()
