from __future__ import annotations

from common.evaluate import auroc, bootstrap_auroc, paired_bootstrap
import numpy as np
import pytest
from scripts.diff_methods import parse_run, sentence


def data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.4
    signal = np.where(correct, 0.0, 1.0)
    noise = rng.normal(size=n)
    good = signal + noise  # informative
    slightly_worse = good + 0.5 * rng.normal(size=n)  # same signal, a little more noise
    return correct, good, slightly_worse, rng.random(n)


def test_identical_methods_are_a_tie_with_zero_difference():
    correct, good, _, _ = data()
    result = paired_bootstrap(good, good, correct, n_boot=200)
    assert result.diff == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.verdict == "tie"


def test_informative_beats_random():
    correct, good, _, random = data()
    result = paired_bootstrap(good, random, correct, n_boot=300)
    assert result.verdict == "A"
    assert result.ci_low > 0
    assert result.value_a == pytest.approx(auroc(good, correct))


def test_swapping_a_and_b_flips_sign_and_verdict():
    correct, good, _, random = data()
    ab = paired_bootstrap(good, random, correct, n_boot=300)
    ba = paired_bootstrap(random, good, correct, n_boot=300)
    assert ba.diff == pytest.approx(-ab.diff)
    assert ba.ci_low == pytest.approx(-ab.ci_high)
    assert ba.verdict == "B"


def test_aurc_difference_is_positive_when_a_is_better():
    correct, good, _, random = data()
    result = paired_bootstrap(good, random, correct, metric="aurc", n_boot=100)
    assert result.value_a < result.value_b  # lower AURC is better
    assert result.diff > 0


def test_paired_interval_is_narrower_than_separate_intervals():
    # The reason for pairing: the two separate AUROC intervals overlap, but the
    # interval of the difference does not contain 0.
    correct, good, slightly_worse, _ = data(n=1000)
    low_a, high_a = bootstrap_auroc(good, correct, n_boot=300)
    _, high_b = bootstrap_auroc(slightly_worse, correct, n_boot=300)
    assert low_a < high_b  # separate intervals overlap
    result = paired_bootstrap(good, slightly_worse, correct, n_boot=300)
    assert result.ci_high - result.ci_low < high_a - low_a
    assert result.verdict == "A"


def test_same_seed_gives_same_interval():
    correct, good, slightly_worse, _ = data()
    first = paired_bootstrap(good, slightly_worse, correct, n_boot=100, seed=3)
    second = paired_bootstrap(good, slightly_worse, correct, n_boot=100, seed=3)
    assert first == second


def test_unknown_metric_and_unequal_lengths_are_rejected():
    correct, good, _, _ = data()
    with pytest.raises(ValueError, match="metric must be one of"):
        paired_bootstrap(good, good, correct, metric="accuracy")
    with pytest.raises(ValueError, match="equally long"):
        paired_bootstrap(good[:-1], good, correct)


def test_parse_run():
    assert parse_run("qwen0.5b/triviaqa_1000") == ("qwen0.5b", "triviaqa", 1000)
    assert parse_run("smollm2-1.7b/mmlu_200") == ("smollm2-1.7b", "mmlu", 200)
    with pytest.raises(ValueError, match="run must look like"):
        parse_run("qwen0.5b")


def test_sentence_names_the_better_method():
    correct, good, _, random = data()
    text = sentence("random", "good", paired_bootstrap(random, good, correct, n_boot=200), "m/d_1")
    assert "good is better than random" in text
    low, high = text.split("advantage [")[1].split("]")[0].split(", ")
    assert float(low) > 0
    assert float(high) > float(low)
    tie = sentence("a", "b", paired_bootstrap(good, good, correct, n_boot=50), "m/d_1")
    assert "cannot be told apart" in tie
