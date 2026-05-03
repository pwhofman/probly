"""Sample multiple LLM answers for each question."""

from __future__ import annotations

import torch

from probly.representer.sampler.huggingface import HFTextGenerationSampler


# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME = "google/gemma-4-E2B-it"


def main() -> None:
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    sampler = HFTextGenerationSampler.from_model_name(
        MODEL_NAME,
        num_samples=2,
        model_kwargs=model_kwargs,
        batch_size=10,
        temperature=1.0,
        max_new_tokens=64,
        top_k=50,
    )

    questions = [
        "What is a quick way to explain semantic entropy?",
        "Name one reason LLM answers can vary across samples.",
    ]
    chats = [[{"role": "user", "content": question}] for question in questions]

    sample = sampler(chats)

    for question, answers, log_likelihoods in zip(
        questions,
        sample.tensor.text,
        sample.tensor.log_likelihood,
        strict=True,
    ):
        print(f"\nQuestion: {question}")
        for index, (answer, log_likelihood) in enumerate(zip(answers, log_likelihoods, strict=True), start=1):
            print(f"\nSample {index} (log likelihood: {log_likelihood.item():.2f})")
            print(answer.strip())


if __name__ == "__main__":
    main()
