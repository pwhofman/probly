"""Cluster sampled LLM answers by semantic equivalence."""

from __future__ import annotations

import logging
import os

import torch

from probly.representer.sampler.huggingface import HFTextGenerationSampler
from probly.representer.semantic_clustering import GreedyHFSemanticClusterer


# GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
NLI_MODEL_NAME = "microsoft/deberta-base-mnli"


def main() -> None:
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    generation_model_kwargs: dict[str, object] = {"dtype": "auto"}
    nli_model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        generation_model_kwargs["device_map"] = "auto"
        nli_model_kwargs["device_map"] = "auto"

    sampler = HFTextGenerationSampler.from_model_name(
        GENERATION_MODEL_NAME,
        num_samples=10,
        model_kwargs=generation_model_kwargs,
        batch_size=4,
        temperature=0.7,
        max_new_tokens=128,
    )
    clusterer = GreedyHFSemanticClusterer.from_model_name(
        NLI_MODEL_NAME,
        model_kwargs=nli_model_kwargs,
        batch_size=10,
    )

    questions = [
        "What is the capital of France?",
        "What is the chemical symbol for water?",
        "How many planets are in our solar system?",
        "Why is the sky blue?",
        "What causes tides in the ocean?",
        "What is the best programming language?",
        "Is a hot dog a sandwich?",
        "Who was the first person to walk on Mars?",
        "What year was the city of Atlantis founded?",
    ]
    chats = [[{"role": "user", "content": question}] for question in questions]

    text_sample = sampler(chats)
    semantic_sample = clusterer(text_sample)

    for question, answers, log_likelihoods, group_ids in zip(
        questions,
        text_sample.tensor.text,
        text_sample.tensor.log_likelihood,
        semantic_sample.group_ids,
        strict=True,
    ):
        print(f"\nQuestion: {question}")
        for index, (answer, log_likelihood, group_id) in enumerate(
            zip(answers, log_likelihoods, group_ids, strict=True),
            start=1,
        ):
            print(
                f"Sample {index} "
                f"(group: {group_id.item()}, log likelihood: {log_likelihood.item():.2f}): "
                f"{answer.strip()[:150]!r}{'...' if len(answer.strip()) > 150 else ''}"
            )


if __name__ == "__main__":
    main()
