"""Tests for the metric flexdispatch fallbacks."""

from __future__ import annotations

import pytest


class TestMetricFallbacks:
    """Each top-level metric raises NotImplementedError for unregistered types."""

    def test_auc_raises(self) -> None:
        from probly.metrics._common import auc  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="auc"):
            auc(object(), object())

    def test_roc_curve_raises(self) -> None:
        from probly.metrics._common import roc_curve  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="roc_curve"):
            roc_curve(object(), object())

    def test_precision_recall_curve_raises(self) -> None:
        from probly.metrics._common import precision_recall_curve  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="precision_recall_curve"):
            precision_recall_curve(object(), object())

    def test_roc_auc_score_raises(self) -> None:
        from probly.metrics._common import roc_auc_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="roc_auc_score"):
            roc_auc_score(object(), object())

    def test_average_precision_score_raises(self) -> None:
        from probly.metrics._common import average_precision_score  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="average_precision_score"):
            average_precision_score(object(), object())

    def test_coverage_raises(self) -> None:
        from probly.metrics._common import coverage  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="coverage"):
            coverage(object(), object())

    def test_efficiency_raises(self) -> None:
        from probly.metrics._common import efficiency  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="efficiency"):
            efficiency(object())

    def test_average_interval_width_raises(self) -> None:
        from probly.metrics._common import average_interval_width  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="average_interval_width"):
            average_interval_width(object())

    def test_convex_hull_coverage_raises(self) -> None:
        from probly.metrics._common import convex_hull_coverage  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="convex_hull_coverage"):
            convex_hull_coverage(object(), object())
