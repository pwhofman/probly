"""Backend-independent contracts for the entropy dispatchers."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

from flextype import Flexdispatch
import pytest

from probly.quantification.measure.credal_set import lower_entropy, upper_entropy
from probly.representation.credal_set import CredalSet


@pytest.mark.parametrize("measure", [upper_entropy, lower_entropy])
def test_entropy_dispatcher_metadata_and_pickle(measure):
    assert isinstance(measure, Flexdispatch)
    assert measure.__name__ in {"upper_entropy", "lower_entropy"}
    assert "entropy of a credal set" in measure.__doc__
    assert list(signature(measure).parameters) == ["credal_set", "base", "return_distribution"]
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
