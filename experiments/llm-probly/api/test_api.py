"""Tests for the uncertainty API (SmolLM2 135M Instruct for the end-to-end test, CPU only)."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
from typing import Any

from fastapi.testclient import TestClient
import pytest
from selector.profile import Calibration, MethodChoice, Metrics, Profile, save_profile

from api.app import app
from api.profiles import find_profile
from api.scoring import (
    Generation,
    generate,
    likelihood_mean,
    likelihood_sum,
    mean_token_entropy,
    perplexity,
)
from common.data import cache_path, load_jsonl

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
METRICS = Metrics(auroc=0.8, auroc_low=0.7, auroc_high=0.9, prr=0.6, prr_low=0.4, prr_high=0.8)
D4_EXAMPLE = Path(__file__).resolve().parent.parent / "docs" / "profile_example_qwen1.5b.json"
client = TestClient(app)


def make_profile(
    name: str = "tiny", method: str = "likelihood_mean", cost: float = 1.0, **changes: Any
) -> Profile:
    """A valid profile for SmolLM2 135M; keyword arguments replace fields."""
    fields: dict[str, Any] = {
        "model": name,
        "model_id": MODEL_ID,
        "dataset": "triviaqa",
        "n_questions": 200,
        "split_seed": 0,
        "calibration_size": 100,
        "method": MethodChoice(name=method, family="baseline", access="logprobs", cost=cost),
        "reason": "test profile",
        "calibration_metrics": METRICS,
        "calibration": Calibration(type="platt", parameters={"a": 1.0, "b": -2.0}),
    }
    fields.update(changes)
    return Profile(**fields)


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A profiles folder with one valid profile named ``tiny``."""
    save_profile(make_profile(), tmp_path / "tiny.json")
    monkeypatch.setenv("PROFILES_DIR", str(tmp_path))
    return tmp_path


def fake_generation(logprob: float) -> Generation:
    """Two tokens with the given log-probability each."""
    return Generation(answer="Paris", token_logprobs=[logprob, logprob], token_entropies=[0.5, 0.5])


def test_end_to_end_with_smollm(profiles: Path) -> None:
    r = client.post("/uncertainty", json={"question": "What is the capital of France?", "model": "tiny"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "confidence", "method", "cost"}
    assert isinstance(body["answer"], str)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["method"] == "likelihood_mean"
    assert body["cost"] == 1.0


def test_model_can_be_given_as_huggingface_id(profiles: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.app.generate", lambda model_id, question: fake_generation(math.log(0.9)))
    r = client.post("/uncertainty", json={"question": "hi", "model": MODEL_ID})
    assert r.status_code == 200
    assert r.json()["method"] == "likelihood_mean"


def test_more_certain_generation_gets_higher_confidence(profiles: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    confidences = []
    for p in (0.95, 0.1):
        monkeypatch.setattr("api.app.generate", lambda model_id, question, p=p: fake_generation(math.log(p)))
        confidences.append(client.post("/uncertainty", json={"question": "hi", "model": "tiny"}).json()["confidence"])
    assert confidences[0] > confidences[1]


def test_unknown_model_returns_404(profiles: Path) -> None:
    r = client.post("/uncertainty", json={"question": "hi", "model": "does-not/exist"})
    assert r.status_code == 404


def test_path_like_model_name_is_not_followed(profiles: Path) -> None:
    assert client.post("/uncertainty", json={"question": "hi", "model": "../tiny"}).status_code == 404


def test_budget_below_method_cost_rejected(profiles: Path) -> None:
    r = client.post("/uncertainty", json={"question": "hi", "model": "tiny", "budget": 0.5})
    assert r.status_code == 422


def test_unsupported_method_returns_501(profiles: Path) -> None:
    save_profile(make_profile(method="p_true_zeroshot"), profiles / "tiny.json")
    r = client.post("/uncertainty", json={"question": "hi", "model": "tiny"})
    assert r.status_code == 501


def test_uncalibrated_raw_score_returns_500(profiles: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.app.generate", lambda model_id, question: fake_generation(math.log(0.9)))
    save_profile(make_profile(calibration=Calibration()), profiles / "tiny.json")
    r = client.post("/uncertainty", json={"question": "hi", "model": "tiny"})
    assert r.status_code == 500


def test_invalid_profile_file_returns_500(profiles: Path) -> None:
    (profiles / "tiny.json").write_text('{"model": "tiny"}')
    r = client.post("/uncertainty", json={"question": "hi", "model": "tiny"})
    assert r.status_code == 500


def test_invalid_file_does_not_hide_valid_profiles(profiles: Path) -> None:
    (profiles / "broken.json").write_text("not json")
    profile = find_profile(MODEL_ID)
    assert profile is not None
    assert profile.model == "tiny"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "model": "tiny"},
        {"question": "hi", "model": ""},
        {"question": "hi"},
        {"question": "hi", "model": "tiny", "budget": -1},
    ],
)
def test_invalid_requests_rejected(profiles: Path, payload: dict) -> None:
    assert client.post("/uncertainty", json=payload).status_code == 422


@pytest.mark.skipif(not D4_EXAMPLE.exists(), reason="D4 example profile not found")
@pytest.mark.parametrize("model", ["qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"])
def test_real_d4_example_profile_is_found_and_not_yet_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """D's example profile loads through the API; its method (p_true_zeroshot) is not computable yet."""
    shutil.copy(D4_EXAMPLE, tmp_path / "qwen1.5b.json")
    monkeypatch.setenv("PROFILES_DIR", str(tmp_path))
    profile = find_profile(model)
    assert profile is not None
    assert profile.method.name == "p_true_zeroshot"
    r = client.post("/uncertainty", json={"question": "hi", "model": model})
    assert r.status_code == 501
    assert "p_true_zeroshot" in r.json()["detail"]


@pytest.mark.skipif(not D4_EXAMPLE.exists(), reason="D4 example profile not found")
def test_real_d4_example_profile_budget_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copy(D4_EXAMPLE, tmp_path / "qwen1.5b.json")
    monkeypatch.setenv("PROFILES_DIR", str(tmp_path))
    r = client.post("/uncertainty", json={"question": "hi", "model": "qwen1.5b", "budget": 1.0})
    assert r.status_code == 422


def test_scorers_on_known_values() -> None:
    gen = fake_generation(math.log(0.5))
    assert likelihood_sum(gen) == pytest.approx(2 * math.log(2))
    assert likelihood_mean(gen) == pytest.approx(math.log(2))
    assert perplexity(gen) == pytest.approx(2.0)
    assert mean_token_entropy(gen) == pytest.approx(0.5)


def test_scorers_handle_empty_generation() -> None:
    empty = Generation(answer="", token_logprobs=[], token_entropies=[])
    assert likelihood_sum(empty) == 0.0
    assert likelihood_mean(empty) == 0.0
    assert perplexity(empty) == 1.0
    assert mean_token_entropy(empty) == 0.0


def test_openapi_docs_available() -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/uncertainty" in r.json()["paths"]


def test_generation_matches_the_benchmark_cache() -> None:
    """The API must reproduce the cached greedy answer and likelihood of the benchmark."""
    path = cache_path("qwen0.5b", 5, "triviaqa")
    if not path.exists():
        pytest.skip(f"{path} not found")
    row = load_jsonl(path)[0]
    gen = generate("Qwen/Qwen2.5-0.5B-Instruct", row["question"])
    assert gen.answer == row["greedy_answer"]
    assert likelihood_mean(gen) == pytest.approx(-row["greedy_ll_mean"], abs=1e-3)
    assert likelihood_sum(gen) == pytest.approx(-row["greedy_ll_sum"], abs=1e-2)
