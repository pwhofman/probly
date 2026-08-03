"""Torch end-to-end tests for the diagnostic suite."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.diagnostics import Verdict, diagnose  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.transformation import dropout  # noqa: E402


@pytest.fixture(scope="module")
def setup() -> dict:
    torch.manual_seed(0)
    inputs = torch.cat([torch.randn(200, 2) + 2.0, torch.randn(200, 2) - 2.0])
    targets = torch.cat([torch.zeros(200, dtype=torch.long), torch.ones(200, dtype=torch.long)])
    ood_inputs = torch.randn(200, 2) * 0.5 + torch.tensor([8.0, -8.0])

    base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.Adam(base.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(base(inputs), targets)
        loss.backward()
        opt.step()

    model = dropout(base, p=0.25, predictor_type="logit_classifier")
    model.eval()
    return {"rep": representer(model, num_samples=30), "x": inputs, "y": targets, "ood": ood_inputs, "baseline": base}


def test_report_structure(setup: dict) -> None:
    with torch.no_grad():
        report = diagnose(setup["rep"], setup["x"], setup["y"], ood_inputs=setup["ood"], baseline=setup["baseline"])
    names = [result.name for result in report.results]
    assert names == [
        "pipeline",
        "accuracy",
        "ece",
        "decomposition_additivity",
        "selective_prediction",
        "ood_separation",
        "baseline_selective_prediction",
    ]
    assert report["pipeline"].verdict is Verdict.PASS
    assert report["accuracy"].verdict is Verdict.INFO
    assert 0.0 <= report["accuracy"].value <= 1.0
    assert report["accuracy"].reference is not None
    assert report["ece"].verdict is Verdict.INFO
    assert 0.0 <= report["ece"].value <= 1.0
    assert report["decomposition_additivity"].verdict is not Verdict.SKIP
    assert report["ood_separation"].value is not None
    assert str(report)


def test_skips_without_optional_inputs(setup: dict) -> None:
    with torch.no_grad():
        report = diagnose(setup["rep"], setup["x"], setup["y"])
    assert report["ood_separation"].verdict is Verdict.SKIP
    assert report["baseline_selective_prediction"].verdict is Verdict.SKIP
    assert report["accuracy"].reference is None


def test_pipeline_failure_is_reported() -> None:
    model = dropout(nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)), predictor_type="logit_classifier")
    rep = representer(model, num_samples=5)
    report = diagnose(rep, torch.randn(10, 5), torch.zeros(10, dtype=torch.long))  # wrong input dimension
    assert report["pipeline"].verdict is Verdict.FAIL
    assert not report.passed
    assert report["selective_prediction"].verdict is Verdict.SKIP
