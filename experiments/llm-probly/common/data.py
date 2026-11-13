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


# Every dataset loader returns rows with ``id``, ``question`` and ``aliases``.
DATASETS: dict[str, Callable[..., list[dict]]] = {
    "triviaqa": load_trivia_questions,
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
