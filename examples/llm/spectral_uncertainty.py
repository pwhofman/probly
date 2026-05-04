"""Compute spectral uncertainty from embedded sampled LLM answers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch

from probly.quantification import decompose
from probly.representer.clarifier.huggingface import HFQuestionClarifier
from probly.representer.embedder.huggingface import HFTextEmbedder
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model


GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NUM_CLARIFICATIONS = 2
NUM_ANSWERS = 5

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
    """Spectral uncertainty results for a single question."""

    category: str
    question: str
    clarifications: list[str]
    responses: list[list[str]]
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float


def format_result(result: QuestionResult, index: int, total: int) -> str:
    """Format one question's spectral uncertainty result for console output."""
    lines = [
        f"=== Question {index}/{total} [{result.category}] ===",
        f"Q: {result.question}",
        "",
        f"Clarifications ({len(result.clarifications)} samples):",
    ]
    for clarification_index, (clarification, responses) in enumerate(
        zip(result.clarifications, result.responses, strict=True),
        start=1,
    ):
        lines.append(f"  Clarification {clarification_index}: {clarification!r}")
        for response_index, response in enumerate(responses, start=1):
            display = response[:180] + "..." if len(response) > 180 else response
            lines.append(f"    {response_index}: {display!r}")

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
    print("\n" + "=" * 74)
    print("Summary")
    print("-" * 74)
    print(f"{'Question':<40} {'TU':>10} {'AU':>10} {'EU':>10}")
    print("-" * 74)
    for result in results:
        question = result.question[:38] + ".." if len(result.question) > 40 else result.question
        print(
            f"{question:<40} {result.total_uncertainty:>10.4f} "
            f"{result.aleatoric_uncertainty:>10.4f} {result.epistemic_uncertainty:>10.4f}"
        )
    print("=" * 74)


def main() -> None:
    """Run spectral uncertainty with input clarification."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    generation_model_kwargs: dict[str, object] = {"dtype": "auto"}
    embedding_model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        generation_model_kwargs["device_map"] = "auto"
        embedding_model_kwargs["device"] = "cuda"

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
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = HFTextEmbedder.from_model_name(
        EMBEDDING_MODEL_NAME,
        model_kwargs=embedding_model_kwargs,
        batch_size=16,
        normalize_embeddings=True,
    )

    questions = [question for _, question in QUESTIONS]
    print(
        f"\nRunning spectral uncertainty with input clarification: {len(QUESTIONS)} questions, "
        f"{clarifier.num_samples} clarifications each, {sampler.num_samples} answers per clarification\n"
    )
    print("Generating clarifications...")
    clarifications = clarifier(questions)
    print("Generating responses...")
    text_sample = sampler(clarifications)
    print("Embedding responses...")
    embeddings = embedder(text_sample)
    print("Computing spectral uncertainty...")
    decomposition = decompose(embeddings)

    results: list[QuestionResult] = []
    for index, (
        (category, question),
        question_clarifications,
        responses,
        total_uncertainty,
        aleatoric_uncertainty,
        epistemic_uncertainty,
    ) in enumerate(
        zip(
            QUESTIONS,
            clarifications.tensor.text,
            text_sample.tensor.tensor.text,
            decomposition.total,
            decomposition.aleatoric,
            decomposition.epistemic,
            strict=True,
        ),
        start=1,
    ):
        result = QuestionResult(
            category=category,
            question=question,
            clarifications=[str(clarification).strip() for clarification in question_clarifications],
            responses=[[str(response).strip() for response in clarification_responses] for clarification_responses in responses],
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
