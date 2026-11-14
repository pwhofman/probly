"""Tests for decomposition lookup and backend type propagation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_TYPING_CHECK = """
from typing import Any, assert_type
from ndlike.array_like import WeakArrayLike

from probly.method.duq import DUQDecomposition, DUQRepresentation, create_duq_representation, duq_uncertainty
from probly.method.mahalanobis import (
    MahalanobisDecomposition,
    MahalanobisRepresentation,
    create_mahalanobis_representation,
)
from probly.method.mahalanobis._common import combine_layer_scores
from probly.quantification.decomposition.decomposition import Decomposition
from probly.quantification.notion import AleatoricUncertainty, Notion
from probly.representation.distribution import CategoricalDistribution

def check_decomposition_lookup(decomposition: Decomposition) -> None:
    assert_type(decomposition[AleatoricUncertainty], AleatoricUncertainty)
    assert_type(decomposition["aleatoric"], Notion)

def check_method_backend_types[T: WeakArrayLike[Any]](
    values: T,
    softmax: CategoricalDistribution[T],
    duq: DUQDecomposition[T],
    mahalanobis: MahalanobisDecomposition[T],
) -> None:
    assert_type(create_duq_representation(values), DUQRepresentation[T])
    assert_type(duq_uncertainty(values), T)
    assert_type(duq.representation, DUQRepresentation[T])
    assert_type(duq.representation.kernel_values, T)
    assert_type(duq.total, T)

    assert_type(create_mahalanobis_representation(softmax, values, values, values), MahalanobisRepresentation[T])
    assert_type(combine_layer_scores(values, values, values), T)
    assert_type(mahalanobis.representation, MahalanobisRepresentation[T])
    assert_type(mahalanobis.representation.softmax, CategoricalDistribution[T])
    assert_type(mahalanobis.representation.layer_scores, T)
    assert_type(mahalanobis.representation.weight, T)
    assert_type(mahalanobis.representation.bias, T)
    assert_type(mahalanobis.epistemic, T)
"""


def test_decomposition_types(tmp_path: Path) -> None:
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for the static decomposition checks")
    source = tmp_path / "check_decomposition.py"
    source.write_text(_TYPING_CHECK, encoding="utf-8")
    # Check source directly: generated partial stubs shadow imports in tests.
    source_root = Path(__file__).resolve().parents[4] / "src"
    result = subprocess.run(  # noqa: S603
        [
            executable,
            "check",
            str(source),
            "--project",
            str(tmp_path),
            "--python",
            sys.prefix,
            "--extra-search-path",
            str(source_root),
            "--error-on-warning",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
