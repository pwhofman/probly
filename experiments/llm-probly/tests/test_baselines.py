"""Tests for the cache-only baselines (BIG): msp, perplexity, mean_token_entropy, lexical_similarity."""

from __future__ import annotations

import math

from common.io import validate_scores
from common.registry import get
from methods import lexical_similarity, likelihood_mean, mean_token_entropy, msp, perplexity
import pytest


def row(i, logprobs, entropies, samples=("Paris",) * 10):
    return {
        "id": i,
        "model": "test/model",
        "greedy_token_logprobs": list(logprobs),
        "greedy_token_entropies": list(entropies),
        "samples": list(samples),
    }


ROWS = [
    row(0, [-0.1, -0.1], [0.2, 0.4]),  # confident
    row(1, [-2.0, -1.0], [1.5, 2.5], samples=["Paris", "Berlin", "a city in France"] + ["Rome"] * 7),  # unsure
    row(2, [], []),  # no scored tokens -> failed
]


@pytest.mark.parametrize("plugin", [msp, perplexity, mean_token_entropy, lexical_similarity])
def test_record_follows_contract(plugin):
    rec = plugin.score(ROWS)
    assert validate_scores(rec, expected_ids=[0, 1, 2]) == []
    assert rec["method"] == plugin.NAME
    assert not get(plugin.NAME).needs_model


def test_values():
    assert msp.score(ROWS)["uncertainty"][:2] == pytest.approx([0.2, 3.0])
    assert perplexity.score(ROWS)["uncertainty"][:2] == pytest.approx([math.exp(0.1), math.exp(1.5)])
    assert mean_token_entropy.score(ROWS)["uncertainty"][:2] == pytest.approx([0.3, 2.0])


@pytest.mark.parametrize("plugin", [msp, perplexity, mean_token_entropy])
def test_unsure_answer_is_more_uncertain(plugin):
    u = plugin.score(ROWS)["uncertainty"]
    assert u[1] > u[0]


@pytest.mark.parametrize("plugin", [msp, perplexity, mean_token_entropy])
def test_empty_answer_is_failed(plugin):
    rec = plugin.score(ROWS)
    assert rec["failed"] == [False, False, True]
    assert rec["uncertainty"][2] == 1.0


def test_perplexity_ranks_like_likelihood_mean():
    rows = [dict(r, greedy_ll_mean=sum(r["greedy_token_logprobs"]) / len(r["greedy_token_logprobs"])) for r in ROWS[:2]]
    a, b = perplexity.score(rows)["uncertainty"], likelihood_mean.score(rows)["uncertainty"]
    assert (a[0] < a[1]) == (b[0] < b[1])


def test_rouge_l():
    assert lexical_similarity.rouge_l("Paris", "paris.") == 1.0
    assert lexical_similarity.rouge_l("Paris", "Berlin") == 0.0
    # LCS of "the city of paris" and "paris city" is 1 word -> P=1/2, R=1/4
    assert lexical_similarity.rouge_l("the city of paris", "paris city") == pytest.approx(2 * 0.5 * 0.25 / 0.75)


def test_identical_samples_are_least_uncertain():
    u = lexical_similarity.score(ROWS)["uncertainty"]
    assert u[0] == pytest.approx(-1.0)
    assert u[1] > u[0]
