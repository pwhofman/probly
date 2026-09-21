"""Backend-independent contracts for generic protected-axis function helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, overload

import pytest

from probly.representation._protected_axis._common_functions import (
    AxisProtectedInternals,
    apply_unary,
    extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
)


def _allowed() -> None:
    pass


def _denied() -> None:
    pass


@dataclass(frozen=True)
class _Value:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class _Rebuilt:
    values: dict[str, _Value]
    func: Callable | None


@dataclass
class _Protected:
    values: dict[str, _Value]
    protected_axes: dict[str, int] = field(default_factory=lambda: {"left": 1, "right": 0})
    permitted_functions: set[Callable[..., Any]] = field(default_factory=lambda: {_allowed})
    read_contexts: list[Callable | None] = field(default_factory=list)

    @overload
    def protected_values(self) -> dict[str, _Value]: ...

    @overload
    def protected_values(self, func: Callable) -> dict[str, _Value] | None: ...

    def protected_values(self, func: Callable | None = None) -> dict[str, _Value] | None:
        self.read_contexts.append(func)
        return self.values if func is None or func in self.permitted_functions else None

    def with_protected_values(self, values: dict[str, _Value], func: Callable | None = None) -> _Rebuilt:
        return _Rebuilt(values, func)


def _extract(
    obj: object, func: Callable | None = None, *, check_is_permitted: bool = False
) -> AxisProtectedInternals[_Value, _Rebuilt] | None:
    if not isinstance(obj, _Protected):
        return None
    return extract_axis_protected_internals(obj, func, check_is_permitted=check_is_permitted)


@pytest.mark.parametrize("check_is_permitted", [False, True])
def test_extraction_preserves_types_context_and_owner(check_is_permitted: bool) -> None:
    obj = _Protected({"left": _Value((2, 3)), "right": _Value((2,))})
    internals = extract_axis_protected_internals(obj, _allowed, check_is_permitted=check_is_permitted)
    assert internals is not None
    assert internals.primary_value is obj.values["left"]
    assert internals.batch_ndim == 1
    assert internals.owner_type is _Protected
    assert internals.values == obj.values
    assert internals.values is not obj.values
    assert internals.protected_axes == obj.protected_axes
    assert internals.protected_axes is not obj.protected_axes
    assert obj.read_contexts == [_allowed if check_is_permitted else None]

    result = apply_unary(
        internals, lambda _name, value, axes: _Value((1, *value.shape[-axes:])) if axes else _Value((1,))
    )
    assert result.values == {"left": _Value((1, 3)), "right": _Value((1,))}
    assert result.func is _allowed


def test_permission_checks_are_optional_but_reconstruction_keeps_context() -> None:
    obj = _Protected({"left": _Value((2, 3)), "right": _Value((2,))})
    assert extract_axis_protected_internals(obj, _denied, check_is_permitted=True) is None
    internals = extract_axis_protected_internals(obj, _denied)
    assert internals is not None
    assert internals.create(internals.values).func is _denied
    assert obj.read_contexts == [_denied, None]

    assert extract_axis_protected_internals(obj, check_is_permitted=True) is not None
    assert obj.read_contexts[-1] is None


@pytest.mark.parametrize("protected_axes", [{}, {"missing": 1}, {"left": 3}])
def test_extraction_rejects_invalid_layout(protected_axes: dict[str, int]) -> None:
    obj = _Protected({"left": _Value((2, 3))}, protected_axes=protected_axes)
    assert extract_axis_protected_internals(obj) is None


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(_Value(()), "removed protected trailing axes"), (_Value((4, 3)), "inconsistent batch-shapes")],
)
def test_unary_validates_fields_before_reconstruction(replacement: _Value, message: str) -> None:
    obj = _Protected({"left": _Value((2, 3)), "right": _Value((2,))})
    internals = extract_axis_protected_internals(obj)
    assert internals is not None
    with pytest.raises(ValueError, match=message):
        apply_unary(internals, lambda name, value, _axes: replacement if name == "left" else value)


def test_sequence_retains_typed_template_and_mixed_operand_order() -> None:
    first = _Protected({"left": _Value((2, 3)), "right": _Value((2,))})
    second = _Protected({"left": _Value((5, 3)), "right": _Value((5,))})
    leading, middle, trailing = object(), object(), object()
    sequence = extract_protected_value_sequence_internals(
        (leading, first, middle, second, trailing), _allowed, extract=_extract
    )
    assert sequence.has_protected
    assert sequence.template is not None
    assert sequence.template.primary_value is first.values["left"]
    for name in first.protected_axes:
        assert sequence.values_by_field[name] == [first.values[name], middle, second.values[name], trailing]
    assert sequence.template.create(first.values).func is _allowed


def test_sequence_without_protected_values_has_no_template() -> None:
    sequence = extract_protected_value_sequence_internals((object(), object()), extract=_extract)
    assert not sequence.has_protected
    assert sequence.template is None
    assert sequence.values_by_field == {}


def test_sequence_rejects_mismatched_layouts() -> None:
    first = _Protected({"left": _Value((2, 3))}, protected_axes={"left": 1})
    second = _Protected({"left": _Value((2, 3))}, protected_axes={"left": 0})
    with pytest.raises(ValueError, match="identical protected_axes"):
        extract_protected_value_sequence_internals((first, second), extract=_extract)


def test_generic_type_contracts(tmp_path: Path) -> None:
    ty = shutil.which("ty")
    if ty is None:
        pytest.skip("ty is available with the lint dependency group")

    # Generated method stubs shadow source imports in the default ty configuration.
    # Check these contracts against source, with unused ignores treated as errors.
    root = Path(__file__).resolve().parents[4]
    config = tmp_path / "ty.toml"
    config.write_text(
        f"[environment]\nroot = ['{root.as_posix()}/src', '{root.as_posix()}']\n"
        "[rules]\nunused-ignore-comment = 'error'\n",
        encoding="utf-8",
    )
    probe = tmp_path / "contracts.py"
    probe.write_text(
        """from typing import assert_type
from probly.representation._protected_axis._common_functions import (
    AxisProtectedInternals, ProtectedValueSequenceInternals,
    apply_unary, extract_axis_protected_internals,
    extract_protected_value_sequence_internals,
)
from tests.probly.representation.protected_axis.test_common import _Protected, _Value, _Rebuilt, _extract

def check(obj: _Protected) -> None:
    internals = extract_axis_protected_internals(obj)
    assert_type(internals, AxisProtectedInternals[_Value, _Rebuilt] | None)
    assert internals is not None
    assert_type(internals.primary_value, _Value)
    assert_type(apply_unary(internals, lambda _name, value, _axes: value), _Rebuilt)
    sequence = extract_protected_value_sequence_internals((obj,), extract=_extract)
    assert_type(sequence, ProtectedValueSequenceInternals[_Value, _Rebuilt])
    assert_type(sequence.template, AxisProtectedInternals[_Value, _Rebuilt] | None)
    apply_unary(internals, lambda _name, _value, _axes: "invalid")  # ty:ignore[invalid-argument-type]
    internals.create({"left": "invalid"})  # ty:ignore[invalid-argument-type]
""",
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603
        [ty, "check", "--config-file", str(config), "--python", sys.executable, "--error-on-warning", str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
