"""Generate diverse clarifications/interpretations of a question.

Used for two-stage sampling in spectral uncertainty decomposition:
first generate clarifications, then generate responses per clarification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

CLARIFICATION_PROMPT = (
    "Rephrase the following question to clarify its meaning. "
    "Provide a single, clear interpretation that differs from the original phrasing. "
    "Do not answer the question, only rephrase it.\n\n"
    "Question: {question}\n\nRephrased question:"
)


def generate_clarifications(
    question: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_clarifications: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate diverse rephrasings/interpretations of a question.

    Each clarification is sampled independently using the same model
    and a rephrasing prompt.

    Args:
        question: The original question to rephrase.
        model: The loaded causal language model.
        tokenizer: The model's tokenizer.
        num_clarifications: Number of clarifications to generate.
        temperature: Sampling temperature (higher = more diverse).
        max_new_tokens: Maximum tokens per clarification.

    Returns:
        List of clarification strings.
    """
    prompt = CLARIFICATION_PROMPT.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    clarifications = []
    for _ in range(num_clarifications):
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        gen_tokens = outputs[0][prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        clarifications.append(text)

    return clarifications
