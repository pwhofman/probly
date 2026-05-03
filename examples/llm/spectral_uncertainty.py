"""Embed sampled LLM answers as a first step toward spectral uncertainty."""

from __future__ import annotations

import logging
import os

import torch

from probly.representer.embedder import HFTextEmbedder
from probly.representer.sampler.huggingface import HFTextGenerationSampler


GENERATION_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
# GENERATION_MODEL_NAME = "google/gemma-4-E2B-it"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

QUESTIONS = [
    "What is the capital of France?",
    "Why is the sky blue?",
]


def main() -> None:
    """Sample LLM responses and print their sentence embeddings."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.getLogger("transformers").setLevel(logging.ERROR)

    generation_model_kwargs: dict[str, object] = {"dtype": "auto"}
    embedding_model_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        generation_model_kwargs["device_map"] = "auto"
        embedding_model_kwargs["device"] = "cuda"

    print(f"Loading generation model: {GENERATION_MODEL_NAME}")
    sampler = HFTextGenerationSampler.from_model_name(
        GENERATION_MODEL_NAME,
        num_samples=2,
        model_kwargs=generation_model_kwargs,
        batch_size=2,
        temperature=0.7,
        max_new_tokens=64,
    )
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = HFTextEmbedder.from_model_name(
        EMBEDDING_MODEL_NAME,
        model_kwargs=embedding_model_kwargs,
        batch_size=8,
        normalize_embeddings=True,
    )

    chats = [[{"role": "user", "content": question}] for question in QUESTIONS]

    print(f"\nGenerating {sampler.num_samples} responses for {len(QUESTIONS)} questions...")
    text_sample = sampler(chats)
    print("Embedding responses...")
    embeddings = embedder(text_sample)

    print(f"\nEmbedding tensor shape: {tuple(embeddings.shape)}")
    for question, responses, response_embeddings in zip(
        QUESTIONS,
        text_sample.tensor.text,
        embeddings.detach().cpu(),
        strict=True,
    ):
        print(f"\nQuestion: {question}")
        for index, (response, embedding) in enumerate(zip(responses, response_embeddings, strict=True), start=1):
            display = str(response).strip()
            display = display[:180] + "..." if len(display) > 180 else display
            print(f"\nSample {index}: {display!r}")
            print(embedding.shape)


if __name__ == "__main__":
    main()
