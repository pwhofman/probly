"""Correctness check for short TriviaQA answers (alias matching).

An answer counts as correct if, after normalisation, any reference alias
appears in it as a whole phrase. Normalisation follows the usual SQuAD /
TriviaQA evaluation: lowercase, drop punctuation and the articles
"a", "an", "the", and collapse whitespace.
"""

from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans(dict.fromkeys(string.punctuation, " "))


def normalize(text: str) -> str:
    """Lowercase, remove punctuation and articles, and collapse whitespace."""
    text = text.lower().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def matching_alias(answer: str, aliases: list[str]) -> str | None:
    """Return the first alias that appears in ``answer`` as a whole phrase, else ``None``.

    Matching on whole words avoids false hits such as the alias "8" inside
    the answer "1822".
    """
    padded_answer = f" {normalize(answer)} "
    for alias in aliases:
        norm_alias = normalize(alias)
        if norm_alias and f" {norm_alias} " in padded_answer:
            return alias
    return None


def is_correct(answer: str, aliases: list[str]) -> bool:
    """Return whether ``answer`` contains any of the reference ``aliases``."""
    return matching_alias(answer, aliases) is not None


_LETTER = re.compile(r"^\W*([A-D])(?![A-Za-z])")
_LETTER_IN_SENTENCE = re.compile(r"\banswer is\W*([A-D])(?![A-Za-z])", re.IGNORECASE)


def answer_letter(answer: str) -> str | None:
    """The option letter of an MMLU answer, else ``None``.

    Accepts answers that start with the letter ("B", "(B)", "B. Paris") and short
    sentences such as "The answer is B."; anything else counts as no answer.
    """
    match = _LETTER.match(answer.strip()) or _LETTER_IN_SENTENCE.search(answer)
    return match.group(1) if match else None


def is_correct_letter(answer: str, aliases: list[str]) -> bool:
    """Exact-letter correctness for multiple choice: the chosen letter equals the gold letter.

    Alias matching cannot be used here, because normalisation removes the article "a",
    which is also the option letter A.
    """
    return answer_letter(answer) == aliases[0].strip().upper()
