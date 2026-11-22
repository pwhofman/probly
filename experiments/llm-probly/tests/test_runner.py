"""Tests for the runner (scripts/run_benchmark.py) and the CLI entry point (cli.py)."""

from __future__ import annotations

import argparse
import json

import cli
from common import io
from common.data import cache_path, load_jsonl
from common.registry import REGISTRY
import pandas as pd
import pytest

from scripts import run_benchmark


def namespace(**kwargs) -> argparse.Namespace:
    defaults = {"methods": None, "skip_model_methods": False, "force": False}
    return argparse.Namespace(**(defaults | kwargs))


def test_find_methods_returns_the_whole_registry():
    assert set(run_benchmark.find_methods()) == set(REGISTRY)


def test_skip_model_methods_drops_self_evaluation():
    chosen = run_benchmark.choose_methods(REGISTRY, {"methods": "all"}, namespace(skip_model_methods=True))
    assert chosen
    assert not any(spec.needs_model for spec in chosen.values())


def test_methods_option_overrides_config():
    chosen = run_benchmark.choose_methods(REGISTRY, {"methods": "all"}, namespace(methods="msp,likelihood_mean"))
    assert list(chosen) == ["msp", "likelihood_mean"]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown methods"):
        run_benchmark.choose_methods(REGISTRY, {"methods": ["msp", "nope"]}, namespace())


def test_family_is_filled_for_methods_skipped_in_this_run():
    # The bug: with --skip-model-methods the p(True) rows came from existing files and had no family.
    table = pd.DataFrame({"method": ["msp", "p_true_zeroshot", "p_true_zeroshot_norm"], "status": ["ok"] * 3})
    out = run_benchmark.add_registry_columns(table)
    assert list(out.columns[:4]) == ["method", "family", "access", "cost"]
    assert out["family"].tolist() == ["baseline", "self_eval", ""]  # last one: unregistered week-1 file
    assert out["cost"].tolist() == ["1", "2", ""]


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(run_benchmark, "RESULTS_ROOT", tmp_path)
    return tmp_path


def test_run_end_to_end_on_the_5_question_cache(tmp_results, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("model: qwen0.5b\ndataset: triviaqa\nn: 5\nprecision: float32\nmethods: all\n")
    # A self-evaluation result from an earlier run with the model: the table must keep its family.
    run_dir = tmp_results / "qwen0.5b" / "triviaqa_5"
    run_dir.mkdir(parents=True)
    rows = load_jsonl(cache_path("qwen0.5b", 5))
    ids = [r["id"] for r in rows]
    earlier = {
        "method": "p_true_zeroshot",
        "model": rows[0]["model"],
        "answer": "greedy",
        "ids": ids,
        "uncertainty": [0.1, 0.9, 0.5, 0.3, 0.7],
        "failed": [False] * 5,
        "seconds_per_question": 0.5,
    }
    (run_dir / "p_true_zeroshot.json").write_text(json.dumps(earlier))

    cli.main(
        [
            "run",
            "--config",
            str(config),
            "--skip-model-methods",
            "--methods",
            "msp,likelihood_mean,dummy_random,p_true_zeroshot",
        ]
    )

    summary = (run_dir / "summary.md").read_text()
    assert "| p_true_zeroshot | self_eval | logprobs | 2 |" in summary
    assert "| msp | baseline | logprobs | 1 |" in summary
    for name in ("msp", "likelihood_mean", "dummy_random"):
        assert (run_dir / f"{name}.json").exists()
    assert "3. p_true_zeroshot" not in capsys.readouterr().out  # skipped: needs the model


def test_cli_methods_filters_by_access(capsys):
    cli.main(["methods", "--access", "text_only", "--json"])
    names = {m["name"] for m in json.loads(capsys.readouterr().out)}
    assert "verbalized_2s" in names
    assert "semantic_entropy" not in names


def test_cli_methods_table_lists_every_method(capsys):
    cli.main(["methods"])
    out = capsys.readouterr().out
    assert all(name in out for name in REGISTRY)
