"""Shared response generation for Gemma experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_responses(
    question: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
) -> tuple[list[str], list[float]]:
    """Generate multiple sampled responses and their log-likelihoods.

    Args:
        question: The question to generate responses for.
        model: The loaded causal language model (AutoModelForCausalLM).
        tokenizer: The model's tokenizer (AutoTokenizer).
        num_samples: Number of independent responses to generate.
        temperature: Sampling temperature (higher = more diverse).
        max_new_tokens: Maximum tokens per generated response.

    Returns:
        Tuple of (response_texts, length_normalized_log_likelihoods).
    """
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    responses = []
    log_likelihoods = []
    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen_tokens = outputs.sequences[0][prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        responses.append(text)

        scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits=True,
        )
        token_log_probs = scores[0].cpu().float()
        num_tokens = (gen_tokens != tokenizer.eos_token_id).sum().item()
        log_lik = token_log_probs[:num_tokens].mean().item() if num_tokens > 0 else 0.0
        log_likelihoods.append(log_lik)

    return responses, log_likelihoods
