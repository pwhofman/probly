"""Backend-agnostic tests for the credal DRO delta schedule."""

from __future__ import annotations

import pytest

from probly.method.credal_dro import credal_dro_deltas


class TestCredalDroDeltas:
    """Uniform interpolation over [delta_g, 1] (Eq. 8 of the CreDRO paper)."""

    def test_matches_paper_example(self) -> None:
        # M=5, delta_g=0.5 is the paper's Table 12 schedule.
        assert credal_dro_deltas(0.5, 5) == pytest.approx([0.5, 0.625, 0.75, 0.875, 1.0])

    def test_single_member_gets_delta_g(self) -> None:
        assert credal_dro_deltas(0.7, 1) == pytest.approx([0.7])

    def test_delta_g_one_reduces_to_plain_ensemble(self) -> None:
        assert credal_dro_deltas(1.0, 3) == pytest.approx([1.0, 1.0, 1.0])

    def test_endpoints_are_float_exact(self) -> None:
        # delta_g + (1 - delta_g) rounds just below 1.0 for e.g. delta_g=0.002; the schedule
        # must pin both ends so the last member takes the CVaR loss's delta == 1 shortcut.
        deltas = credal_dro_deltas(0.002, 10)
        assert deltas[0] == 0.002
        assert deltas[-1] == 1.0

    def test_delta_g_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_g"):
            credal_dro_deltas(0.0, 5)
        with pytest.raises(ValueError, match="delta_g"):
            credal_dro_deltas(1.5, 5)

    def test_num_members_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="num_members"):
            credal_dro_deltas(0.5, 0)
