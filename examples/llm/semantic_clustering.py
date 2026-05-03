"""Cluster sampled LLM answers by semantic equivalence."""

from __future__ import annotations

import os

import torch

from probly.representer.sampler.huggingface import HFTextGenerationSampler
from probly.representer.semantic_clustering import GreedyHFSemanticClusterer


GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
NLI_MODEL_NAME = "microsoft/deberta-base-mnli"


def main() -> None:
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    generation_model_kwargs: dict[str, object] = {"dtype": "auto"}
    nli_model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        generation_model_kwargs["device_map"] = "auto"
        nli_model_kwargs["device_map"] = "auto"

    sampler = HFTextGenerationSampler.from_model_name(
        GENERATION_MODEL_NAME,
        num_samples=4,
        model_kwargs=generation_model_kwargs,
        batch_size=4,
        temperature=1.0,
        max_new_tokens=64,
        top_k=50,
    )
    clusterer = GreedyHFSemanticClusterer.from_model_name(
        NLI_MODEL_NAME,
        model_kwargs=nli_model_kwargs,
        batch_size=16,
    )

    questions = [
        "What is a quick way to explain semantic entropy?",
        "Name one reason LLM answers can vary across samples.",
    ]
    chats = [[{"role": "user", "content": question}] for question in questions]

    text_sample = sampler(chats)
    semantic_sample = clusterer(text_sample)

    for question, answers, log_likelihoods, cluster_ids in zip(
        questions,
        text_sample.tensor.text,
        text_sample.tensor.log_likelihood,
        semantic_sample.tensor.cluster_id,
        strict=True,
    ):
        print(f"\nQuestion: {question}")
        for index, (answer, log_likelihood, cluster_id) in enumerate(
            zip(answers, log_likelihoods, cluster_ids, strict=True),
            start=1,
        ):
            print(
                f"\nSample {index} "
                f"(cluster: {cluster_id.item()}, log likelihood: {log_likelihood.item():.2f})"
            )
            print(answer.strip())


if __name__ == "__main__":
    main()
