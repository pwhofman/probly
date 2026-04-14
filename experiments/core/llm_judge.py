"""LLM-based correctness judging via the Anthropic API.

Provides a general-purpose binary correctness judge that can be reused
across experiments. Uses Claude to determine whether a response correctly
answers a question, given reference answers for comparison.
"""

from __future__ import annotations

import json
from typing import Literal

import anthropic

JudgeModel = Literal["claude-sonnet-4-6-20250514"]

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether a response correctly "
    "answers a question. You will be given the question, a candidate response, "
    "and a list of accepted reference answers.\n\n"
    "Judge whether the candidate response provides a correct answer to the "
    "question. The response does not need to match the reference answers "
    "exactly -- it just needs to be factually correct and answer the question.\n\n"
    'You MUST respond with exactly this JSON and nothing else: {"correct": true} '
    'or {"correct": false}'
)


def judge_correctness(
    question: str,
    response: str,
    reference_answers: list[str],
    model: JudgeModel = "claude-sonnet-4-6-20250514",
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
    client: anthropic.Anthropic | None = None,
) -> bool:
    """Ask an LLM whether a response correctly answers a question.

    Args:
        question: The question that was asked.
        response: The candidate response to evaluate.
        reference_answers: Accepted correct answers for comparison.
        model: Which Claude model to use as judge.
        system_prompt: System prompt for the judge. Override to tailor
            for specific domains (e.g. math, coding).
        client: Anthropic client to reuse. Created automatically if not
            provided.
    """
    if client is None:
        client = anthropic.Anthropic()

    refs = "\n".join(f"- {a}" for a in reference_answers)
    user_message = f"Question: {question}\n\nCandidate response: {response}\n\nAccepted reference answers:\n{refs}"

    result = client.messages.create(
        model=model,
        max_tokens=32,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    text = result.content[0].text.strip()
    try:
        parsed = json.loads(text)
        return bool(parsed["correct"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        msg = f"LLM judge returned unparseable response: {text!r}"
        raise ValueError(msg) from exc


def judge_correctness_batch(
    items: list[tuple[str, str, list[str]]],
    model: JudgeModel = "claude-sonnet-4-6-20250514",
    system_prompt: str = DEFAULT_JUDGE_SYSTEM_PROMPT,
    client: anthropic.Anthropic | None = None,
) -> list[bool]:
    """Judge correctness for a batch of (question, response, references).

    Processes items sequentially. For rate-limit handling, the anthropic
    SDK retries automatically on 429 responses.

    Args:
        items: List of (question, response, reference_answers) tuples.
        model: Which Claude model to use as judge.
        system_prompt: System prompt for the judge.
        client: Anthropic client to reuse across calls.
    """
    if client is None:
        client = anthropic.Anthropic()
    return [
        judge_correctness(question, response, refs, model, system_prompt, client) for question, response, refs in items
    ]
