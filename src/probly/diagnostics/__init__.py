"""Diagnostics for checking that an uncertainty method behaves meaningfully."""

from __future__ import annotations

from ._diagnose import diagnose
from ._report import DiagnosticReport, DiagnosticResult, Verdict

__all__ = [
    "DiagnosticReport",
    "DiagnosticResult",
    "Verdict",
    "diagnose",
]
