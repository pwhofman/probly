"""Tests for the core quantification dispatch fallbacks."""

from __future__ import annotations

import pytest


class TestQuantificationFallbacks:
    """Each top-level dispatch raises NotImplementedError for unregistered types."""

    def test_measure_atomic_raises(self) -> None:
        from probly.quantification._quantification import measure_atomic  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="measure_atomic"):
            measure_atomic(object())

    def test_decompose_raises_for_unknown_type(self) -> None:
        from probly.quantification._quantification import decompose  # noqa: PLC0415

        # Object isn't a Representation; both decompose and measure dispatch fail.
        with pytest.raises(NotImplementedError):
            decompose(object())

    def test_measure_falls_back_to_measure_atomic_when_no_decompose(self) -> None:
        # Build a fake Representation type with no decompose handler — measure should
        # fall back to measure_atomic, which itself raises.
        from probly.quantification._quantification import measure  # noqa: PLC0415
        from probly.representation.representation import Representation  # noqa: PLC0415

        class _FakeRep(Representation):
            pass

        with pytest.raises(NotImplementedError):
            measure(_FakeRep())

    def test_quantify_raises_when_no_handler(self) -> None:
        from probly.quantification._quantification import quantify  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="quantify"):
            quantify(object())
