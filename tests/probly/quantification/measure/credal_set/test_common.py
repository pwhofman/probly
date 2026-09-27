"""Backend-independent contracts for the entropy dispatchers."""

from __future__ import annotations

from inspect import signature
import itertools
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from flextype import Flexdispatch
import numpy as np
import pytest

from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.credal_set import CredalSet

from ._entropy_suite import (
    entropy,
    interval_min_entropy,
    interval_min_entropy_by_free_class,
    previous_greedy_min_entropy,
    random_intervals,
)


@pytest.mark.parametrize("measure", [upper_entropy, lower_entropy])
def test_entropy_dispatcher_metadata_and_pickle(measure):
    assert isinstance(measure, Flexdispatch)
    assert measure.__name__ in {"upper_entropy", "lower_entropy"}
    assert "entropy of a credal set" in measure.__doc__
    parameters = ["credal_set", "base", "return_distribution"]
    if measure is lower_entropy:
        parameters.append("approximate")
        assert signature(measure).parameters["approximate"].default is False
    assert list(signature(measure).parameters) == parameters
    assert signature(measure).parameters["return_distribution"].default is False
    assert pickle.loads(pickle.dumps(measure)) is measure  # noqa: S301


@pytest.mark.parametrize("measure", [upper_entropy, lower_entropy])
def test_entropy_dispatcher_unsupported_type(measure):
    with pytest.raises(NotImplementedError, match="not supported for credal sets"):
        measure(CredalSet())


@pytest.mark.parametrize("measure", [upper_entropy, lower_entropy])
@pytest.mark.parametrize("delayed", [False, True])
def test_entropy_registration_preserves_call_arguments(measure, delayed):
    class CustomCredalSet(CredalSet):
        pass

    value = CustomCredalSet()
    entropy, distribution = object(), object()
    calls = []
    loaded_types = []

    def handler(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return (entropy, distribution) if kwargs.get("return_distribution", False) else entropy

    if delayed:

        @measure.delayed_register(CustomCredalSet)
        def load_backend(cls):
            loaded_types.append(cls)
            measure.register(cls, handler)

    else:
        assert measure.register(CustomCredalSet)(handler) is handler

    assert loaded_types == []
    assert measure(value) is entropy
    assert measure(value, 2.0, return_distribution=True) == (entropy, distribution)
    assert measure(value, base="normalize", return_distribution=False) is entropy
    assert calls == [
        ((value,), {}),
        ((value, 2.0), {"return_distribution": True}),
        ((value,), {"base": "normalize", "return_distribution": False}),
    ]
    assert loaded_types == ([CustomCredalSet] if delayed else [])
    assert measure.dispatch(CustomCredalSet) is handler


_TYPING_CHECK = """
from typing import assert_type

from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.array_like import ArrayLike
from probly.representation.credal_set import CredalSet

def load_backend(cls: type) -> None:
    pass

def check(credal_set: CredalSet, flag: bool, result: ArrayLike) -> None:
    def handler(value: CredalSet) -> ArrayLike:
        return result

    for measure in (upper_entropy, lower_entropy):
        assert_type(measure(credal_set), ArrayLike)
        assert_type(measure(credal_set, base="normalize"), ArrayLike)
        assert_type(measure(credal_set, return_distribution=False), ArrayLike)
        assert_type(measure(credal_set, 2.0, return_distribution=True), tuple[ArrayLike, ArrayLike])
        assert_type(measure(credal_set, return_distribution=flag), ArrayLike | tuple[ArrayLike, ArrayLike])
        measure.register(CredalSet, handler)
        measure.register(CredalSet)(handler)
        measure.delayed_register(CredalSet, load_backend)
        measure.delayed_register((CredalSet, "some.module.CredalSet"))(load_backend)
        measure.dispatch(CredalSet)

        # Unused ignores catch signatures that accidentally become too broad.
        measure(credal_set, base="invalid")  # ty: ignore[invalid-argument-type]
        measure(credal_set, return_distribution="invalid")  # ty: ignore[no-matching-overload]
        measure(credal_set, unsupported=True)  # ty: ignore[no-matching-overload]
"""


def test_entropy_dispatcher_types(tmp_path: Path):
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for the static dispatcher checks")
    source = tmp_path / "check_entropy.py"
    source.write_text(_TYPING_CHECK, encoding="utf-8")
    # Check source directly: generated partial stubs shadow imports in tests.
    source_root = Path(__file__).resolve().parents[5] / "src"
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


def _all_fill_orders_min_entropy(lower: np.ndarray, upper: np.ndarray) -> float:
    """Minimum entropy over the greedy fills in every order, which reach every vertex."""
    best = np.inf
    for order in itertools.permutations(range(len(lower))):
        p = lower.astype(np.float64)
        remaining = 1.0 - p.sum()
        for i in order:
            fill = min(max(remaining, 0.0), upper[i] - lower[i])
            p[i] += fill
            remaining -= fill
        best = min(best, float(entropy(p)))
    return best


@pytest.mark.parametrize("n_classes", range(1, 7))
@pytest.mark.parametrize("kind", ["ensemble", "dirichlet", "sparse", "box"])
def test_interval_references_agree(n_classes: int, kind: str) -> None:
    """The three exact interval references agree, and the previous greedy never beats them."""
    rng = np.random.default_rng(n_classes)
    for _ in range(5):
        lower, upper = random_intervals(rng, kind, n_classes)
        exact = interval_min_entropy(lower, upper)
        np.testing.assert_allclose(interval_min_entropy_by_free_class(lower, upper), exact, atol=1e-12)
        np.testing.assert_allclose(_all_fill_orders_min_entropy(lower, upper), exact, atol=1e-12)
        assert previous_greedy_min_entropy(lower, upper) >= exact - 1e-12
