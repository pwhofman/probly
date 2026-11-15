"""Shared data helpers for the LLM-probly experiments.

Adapted from ``experiments/gemma/experiments/calibration/generate.py``
(``load_trivia_questions``) so that every team member loads exactly the
same TriviaQA questions in the same order.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

DATASET_NAME = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc.nocontext"
DATASET_SPLIT = "validation"
DEFAULT_SEED = 42

# experiments/llm-probly/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_trivia_questions(num_questions: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Sample questions from the TriviaQA validation split.

    The same ``seed`` always gives the same questions in the same order, so
    the 5-question file is the first 5 questions of the 50-question file.

    Args:
        num_questions: How many questions to return.
        seed: Random seed for the shuffle.

    Returns:
        List of dicts with keys ``id``, ``question`` and ``aliases``. The
        canonical answer is always the first alias.
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)  # noqa: S311 (not used for security)

    questions = []
    for qid, idx in enumerate(indices[:num_questions]):
        item = ds[idx]
        value = item["answer"]["value"]
        aliases = [value, *(a for a in item["answer"]["aliases"] if a != value)]
        questions.append({"id": qid, "question": item["question"], "aliases": aliases})
    return questions


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    """Write one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


MMLU_NAME = "LM-Polygraph/mmlu"
MMLU_CONFIG = "continuation"  # 5-shot prompt per subject, at most 100 test questions per subject
MMLU_SPLIT = "test"
# Chat models otherwise start explaining ("Let's analyze each option...") and never reach a letter.
MMLU_INSTRUCTION = "Reply with only the letter (A, B, C or D) of the correct answer."


def load_mmlu_questions(num_questions: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Sample multiple-choice questions from LM-Polygraph's prepared MMLU.

    LM-Polygraph's ``continuation`` config stores, per question, a ready prompt
    (``input``: subject description, five solved examples, then the question
    with options A-D and an empty ``Answer:``) and the gold letter (``output``).
    That prompt is meant for base models, which simply continue with a letter.
    Instruct models treat it as a chat message and start explaining, so we put a
    one-line instruction (``MMLU_INSTRUCTION``) in front and otherwise keep it unchanged.

    Returns:
        Dicts with ``id``, ``question`` (only the last question block, for reading),
        ``prompt`` (the full prompt that is sent) and ``aliases`` (``[gold letter]``).
    """
    ds = load_dataset(MMLU_NAME, MMLU_CONFIG, split=MMLU_SPLIT)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)  # noqa: S311 (not used for security)

    questions = []
    for qid, idx in enumerate(indices[:num_questions]):
        item = ds[idx]
        question = item["input"].rsplit("Q:", 1)[-1].removesuffix("Answer:").strip()
        prompt = f"{MMLU_INSTRUCTION}\n\n{item['input']}"
        questions.append({"id": qid, "question": question, "prompt": prompt, "aliases": [item["output"].strip()]})
    return questions


# Every dataset loader returns rows with ``id``, ``question`` and ``aliases``; a loader
# may add ``prompt`` when the dataset brings its own prompt (then it is sent as is).
DATASETS: dict[str, Callable[..., list[dict]]] = {
    "triviaqa": load_trivia_questions,
    "mmlu": load_mmlu_questions,
}


def load_questions(dataset: str, num_questions: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Load ``num_questions`` questions of a registered dataset."""
    if dataset not in DATASETS:
        msg = f"Unknown dataset {dataset!r}; known: {sorted(DATASETS)}"
        raise ValueError(msg)
    return DATASETS[dataset](num_questions, seed=seed)


def cache_path(model: str, n: int, dataset: str = "triviaqa", suffix: str = ".jsonl") -> Path:
    """Path of a cache file, e.g. ``data/qwen0.5b/triviaqa_200.jsonl`` (``suffix=".meta.json"`` for the meta file)."""
    return DATA_DIR / model / f"{dataset}_{n}{suffix}"


def load_data(model: str = "qwen0.5b", n: int = 50, dataset: str = "triviaqa") -> list[dict]:
    """Load the shared generated data, e.g. ``load_data("qwen0.5b", 200)``.

    Args:
        model: Sub-folder of ``data/`` (``"qwen0.5b"``, ``"qwen1.5b"``, ``"qwen3b"``, ...).
        n: Number of questions (``5``, ``50`` or ``200``).
        dataset: Dataset name (``"triviaqa"``).

    Returns:
        One dict per question, in ``id`` order.
    """
    return load_jsonl(cache_path(model, n, dataset))
