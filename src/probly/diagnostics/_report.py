"""Report objects for uncertainty method diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Outcome of a single diagnostic."""

    PASS = "pass"  # noqa: S105
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of a single diagnostic."""

    name: str
    verdict: Verdict
    value: float | None = None
    reference: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class DiagnosticReport:
    """Collection of diagnostic results with a table rendering."""

    results: tuple[DiagnosticResult, ...]

    @property
    def passed(self) -> bool:
        """True if no diagnostic failed."""
        return all(result.verdict is not Verdict.FAIL for result in self.results)

    def __getitem__(self, name: str) -> DiagnosticResult:
        """Get a result by diagnostic name."""
        for result in self.results:
            if result.name == name:
                return result
        msg = f"No diagnostic named {name!r} in this report."
        raise KeyError(msg)

    def __str__(self) -> str:
        """Render the report as an aligned text table."""
        fmt = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
        rows = [("diagnostic", "verdict", "value", "reference", "detail")]
        rows += [(r.name, r.verdict.value.upper(), fmt(r.value), fmt(r.reference), r.detail) for r in self.results]
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        return "\n".join(
            "  ".join(col.ljust(w) for col, w in zip(row[:4], widths, strict=True)) + "  " + row[4] for row in rows
        )
