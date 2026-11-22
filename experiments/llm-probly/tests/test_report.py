"""Tests for the benchmark report (E4, scripts/report.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import report


def table(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Minimal evaluated table: method, AUROC, vs_best."""
    return pd.DataFrame(
        [
            {
                "method": m,
                "family": "baseline",
                "access": "logprobs",
                "cost": "1",
                "auroc": a,
                "ci_low": a - 0.05,
                "ci_high": a + 0.05,
                "vs_best": v,
                "d_auroc": 0.0,
                "d_ci_low": 0.0,
                "d_ci_high": 0.0,
                "aurc": 0.3,
                "prr": 0.4,
                "ece": float("nan"),
                "seconds_per_question": 0.0,
            }
            for m, a, v in rows
        ]
    )


def test_pick_is_the_cheapest_tied_method():
    t = table([("p_true_zeroshot", 0.87, "best"), ("msp", 0.82, "tied"), ("semantic_entropy", 0.73, "worse")])
    assert report.pick_method(t) == "msp"  # cost 1 beats cost 2


def test_higher_auroc_breaks_a_cost_tie():
    t = table([("msp", 0.82, "tied"), ("mean_token_entropy", 0.83, "best"), ("likelihood_mean", 0.81, "tied")])
    assert report.pick_method(t) == "mean_token_entropy"


def test_api_pick_skips_white_box_methods():
    t = table([("mean_token_entropy", 0.83, "best"), ("msp", 0.82, "tied")])
    assert report.pick_method(t, report.API_ACCESS) == "msp"


def test_api_pick_says_so_when_only_white_box_is_tied():
    t = table([("mean_token_entropy", 0.83, "best"), ("msp", 0.70, "worse")])
    assert report.pick_method(t, report.API_ACCESS) == report.NOT_TIED


def test_random_reference_is_never_the_pick():
    t = table([("p_true_zeroshot", 0.80, "best"), ("dummy_random", 0.55, "worse"), ("msp", 0.78, "tied")])
    assert report.pick_method(t) == "msp"


def test_no_signal_when_random_is_tied_with_the_best():
    # 1.5B on MMLU: everything is near 0.5, so the random reference cannot be told apart from the best.
    t = table([("msp", 0.55, "best"), ("dummy_random", 0.50, "tied"), ("likelihood_mean", 0.51, "tied")])
    assert report.pick_method(t) == report.NO_SIGNAL


def test_model_order_puts_qwen_sizes_first():
    models = ["smollm2-1.7b", "qwen7b", "qwen0.5b", "qwen3b"]
    assert sorted(models, key=report.model_key) == ["qwen0.5b", "qwen3b", "qwen7b", "smollm2-1.7b"]


def test_find_runs_skips_smoke_tests_and_runs_without_cache(tmp_path, monkeypatch):
    for run in ("qwen0.5b/triviaqa_200", "qwen0.5b/triviaqa_5", "qwen9b/triviaqa_200", "qwen0.5b/plots_x"):
        (tmp_path / run).mkdir(parents=True)
        (tmp_path / run / "msp.json").write_text(json.dumps({}))
    monkeypatch.setattr(report, "cache_path", lambda model, n, dataset: tmp_path / f"{model}{n}{dataset}")
    (tmp_path / "qwen0.5b200triviaqa").touch()  # only this run has a data cache
    assert report.find_runs(tmp_path) == [("qwen0.5b", "triviaqa", 200)]


@pytest.fixture
def runs() -> list[report.RunReport]:
    t = table([("p_true_zeroshot", 0.87, "best"), ("msp", 0.82, "tied"), ("dummy_random", 0.5, "worse")])
    return [report.RunReport("qwen7b", "triviaqa", 200, 0.59, "float16, mps", t, "msp", "msp")]


def test_markdown_has_overview_comparison_and_run_table(runs):
    text = report.to_markdown(runs, n_boot=100)
    assert "## Runs" in text
    assert "## Methods x models, triviaqa 200" in text
    assert "## qwen7b/triviaqa_200" in text
    assert "| msp ★ |" in text  # the pick is marked
    assert "**0.870**" in text  # best per model in bold


def test_html_is_one_page_with_escaped_cells(runs):
    runs[0].settings = "<script>x</script>"
    page = report.to_html(runs, n_boot=100)
    assert page.startswith("<!DOCTYPE html>")
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;" in page
    assert 'class="tied pick"' in page
