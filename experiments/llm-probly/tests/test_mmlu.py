"""Tests for the MMLU loader, exact-letter correctness and the prompt choice."""

from __future__ import annotations

import pytest

import common.data
from common.correctness import answer_letter, is_correct_letter

PROMPT = (
    "The following are multiple choice questions (with answers) about astronomy.\n\n"
    "Q:What is the closest star?\nA. Sirius\nB. The Sun\nC. Vega\nD. Polaris\nAnswer:B\n"
    "Q:Which planet is largest?\nA. Mars\nB. Earth\nC. Jupiter\nD. Venus\nAnswer:"
)


@pytest.mark.parametrize(
    ("answer", "letter"),
    [
        ("C", "C"),
        ("C.", "C"),
        ("(C)", "C"),
        ("C. Jupiter", "C"),
        ("The answer is C.", "C"),
        ("The correct answer is (C) Jupiter", "C"),
        (" c", None),
        ("Jupiter", None),
        ("Answer", None),
        ("Let's analyze each option step-by-step:", None),
    ],
)
def test_answer_letter(answer: str, letter: str | None) -> None:
    assert answer_letter(answer) == letter


def test_letter_a_is_not_lost_like_an_article() -> None:
    assert is_correct_letter("A. Sirius", ["A"])
    assert not is_correct_letter("B", ["A"])


def test_mmlu_loader_keeps_prompt_and_gold_letter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = [{"input": PROMPT, "output": "C"}, {"input": PROMPT.replace("largest", "smallest"), "output": "A"}]
    monkeypatch.setattr(common.data, "load_dataset", lambda *_args, **_kwargs: fake)

    rows = common.data.load_questions("mmlu", 2, seed=0)

    assert {r["aliases"][0] for r in rows} == {"C", "A"}
    for r in rows:
        assert r["prompt"].startswith(common.data.MMLU_INSTRUCTION)
        assert r["prompt"].removeprefix(common.data.MMLU_INSTRUCTION + "\n\n") in {f["input"] for f in fake}
        assert r["question"].startswith("Which planet")
        assert r["question"].endswith("D. Venus")


def test_generate_sends_dataset_prompt_as_built() -> None:
    pytest.importorskip("torch")
    from scripts.generate import INSTRUCTION, build_prompt  # noqa: PLC0415

    assert build_prompt({"question": "Q", "prompt": PROMPT}) == PROMPT
    assert build_prompt({"question": "Who?"}) == f"{INSTRUCTION}\nQuestion: Who?"
