"""Tests for the alias-matching correctness check."""

from __future__ import annotations

from common.correctness import is_correct, normalize


def test_normalize_drops_case_punctuation_and_articles() -> None:
    assert normalize("  The Pacific Ocean! ") == "pacific ocean"


def test_exact_and_contained_answers_are_correct() -> None:
    assert is_correct("Thomas Cranmer", ["Thomas Cranmer"])
    assert is_correct("marram grass, fescue, and rye are types of grass.", ["Grass", "Types of grass"])
    assert is_correct("1822", ["1822", "one thousand, eight hundred and twenty-two"])


def test_wrong_answers_are_not_correct() -> None:
    assert not is_correct("United Kingdom", ["Canada", "Dominion of Canada"])
    assert not is_correct("Thickening.", ["Reduction", "Reductively"])


def test_alias_must_match_whole_words() -> None:
    assert not is_correct("1822", ["8"])
    assert not is_correct("Canadian", ["Canada"])


def test_empty_alias_never_matches() -> None:
    assert not is_correct("anything", ["The", "!"])
