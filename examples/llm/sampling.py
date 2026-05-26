"""Sample question clarifications and answers with an LLM."""

from __future__ import annotations

import torch

from probly.representer.clarifier.huggingface import HFQuestionClarifier
from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model


# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME = "google/gemma-4-E2B-it"


def main() -> None:
    model_kwargs: dict[str, object] = {"dtype": "auto"}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model, tokenizer = load_model(MODEL_NAME, model_kwargs=model_kwargs)
    model.eval()

    clarifier = HFQuestionClarifier(
        model=model,
        tokenizer=tokenizer,
        num_samples=2,
        batch_size=10,
        temperature=1.0,
        max_new_tokens=48,
        top_k=50,
    )
    answer_sampler = HFTextGenerationSampler(
        model=model,
        tokenizer=tokenizer,
        num_samples=2,
        batch_size=10,
        temperature=1.0,
        max_new_tokens=64,
        top_k=50,
    )

    questions = [
        "What is a quick way to explain semantic entropy?",
        "Name one reason LLM answers can vary across samples.",
    ]
    clarifications = clarifier(questions)
    answers = answer_sampler(clarifications)

    for question, question_clarifications, question_answers, question_answer_log_likelihoods in zip(
        questions,
        clarifications.tensor.text,
        answers.tensor.tensor.text,
        answers.tensor.tensor.log_likelihood,
        strict=True,
    ):
        print(f"\nQuestion: {question}")
        for clarification_index, (clarification, clarification_answers, answer_log_likelihoods) in enumerate(
            zip(question_clarifications, question_answers, question_answer_log_likelihoods, strict=True),
            start=1,
        ):
            print(f"\nClarification {clarification_index}: {clarification.strip()}")
            for answer_index, (answer, log_likelihood) in enumerate(
                zip(clarification_answers, answer_log_likelihoods, strict=True),
                start=1,
            ):
                print(f"\nAnswer {answer_index} (log likelihood: {log_likelihood.item():.2f})")
                print(answer.strip())


if __name__ == "__main__":
    main()
