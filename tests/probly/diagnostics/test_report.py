"""Backend-free tests for the diagnostic report objects."""

from __future__ import annotations

import pytest

from probly.diagnostics import DiagnosticReport, DiagnosticResult, Verdict


@pytest.fixture
def report() -> DiagnosticReport:
    return DiagnosticReport(
        (
            DiagnosticResult(name="pipeline", verdict=Verdict.PASS, detail="ok"),
            DiagnosticResult(name="selective_prediction", verdict=Verdict.WARN, value=0.1, reference=0.11),
            DiagnosticResult(name="ood_separation", verdict=Verdict.SKIP, detail="no ood_inputs given"),
        )
    )


def test_passed_without_failures(report: DiagnosticReport) -> None:
    assert report.passed


def test_passed_is_false_with_failure() -> None:
    report = DiagnosticReport((DiagnosticResult(name="pipeline", verdict=Verdict.FAIL),))
    assert not report.passed


def test_getitem_by_name(report: DiagnosticReport) -> None:
    assert report["selective_prediction"].value == 0.1
    with pytest.raises(KeyError):
        report["unknown"]


def test_str_renders_all_rows(report: DiagnosticReport) -> None:
    rendered = str(report)
    for name in ("pipeline", "selective_prediction", "ood_separation"):
        assert name in rendered
    assert "PASS" in rendered
    assert "WARN" in rendered
    assert "0.1000" in rendered
