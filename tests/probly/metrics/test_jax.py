"""JAX backend tests for probly.metrics."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ._ap_score_suite import APScoreSuite  # noqa: E402
from ._auc_suite import AUCSuite  # noqa: E402
from ._pr_curve_suite import PRCurveSuite  # noqa: E402
from ._reference_suite import ReferenceSuite  # noqa: E402
from ._roc_auc_score_suite import RocAucScoreSuite  # noqa: E402
from ._roc_curve_suite import RocCurveSuite  # noqa: E402


@pytest.fixture
def array_fn():
    return jnp.array


@pytest.fixture
def array_type():
    return jax.Array


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
