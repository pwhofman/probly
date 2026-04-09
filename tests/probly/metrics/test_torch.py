"""PyTorch backend tests for probly.metrics."""

from __future__ import annotations

import pytest

from ._ap_score_suite import APScoreSuite
from ._auc_suite import AUCSuite
from ._pr_curve_suite import PRCurveSuite
from ._reference_suite import ReferenceSuite
from ._roc_auc_score_suite import RocAucScoreSuite
from ._roc_curve_suite import RocCurveSuite

torch = pytest.importorskip("torch")


@pytest.fixture
def array_fn():
    return torch.tensor


@pytest.fixture
def array_type():
    return torch.Tensor


class TestAUC(AUCSuite):
    pass


class TestRocCurve(RocCurveSuite):
    pass


class TestPRCurve(PRCurveSuite):
    pass


class TestRocAucScore(RocAucScoreSuite):
    pass


class TestAPScore(APScoreSuite):
    pass


class TestReference(ReferenceSuite):
    pass
