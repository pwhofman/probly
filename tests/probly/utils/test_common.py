from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from probly.utils.switchdispatch import switch, switchdispatch


class Base:
    pass


class Derived(Base):
    pass


def test_value_registration_preserves_subtype_and_identity():
    registry = switch[str, Base]()
    value = Derived()

    assert registry.register("direct", value) is value
    assert registry.multi_register(["first", "second"], value) is value
    assert registry.register("decorated")(value) is value
    assert registry.multi_register(["third", "fourth"])(value) is value
    assert registry.register("keyword")(value=value) is value
    assert all(registered is value for registered in registry.values())


def test_class_registration_preserves_generic_class():
    registry = switch[str, type[Base]]()

    @registry.register("generic")
    @registry.multi_register(["first", "second"])
    class GenericValue[T](Base):
        def __init__(self, value: T) -> None:
            self.value = value

    direct = registry.register("direct", GenericValue)
    multiple = registry.multi_register(["third", "fourth"], GenericValue)
    assert GenericValue[float](1.0).value == 1.0
    assert direct[float](1.0).value == 1.0
    assert multiple[float](1.0).value == 1.0
    assert direct is multiple is GenericValue
    assert all(registered is GenericValue for registered in registry.values())


class Handler:
    label = "handler"

    def __call__(self, key: str, value: int) -> int:
        return len(key) + value


def test_function_registration_preserves_callable_object():
    @switchdispatch
    def dispatch(key: str, value: int) -> int:
        return value - len(key)

    handler = Handler()
    assert dispatch.register("direct", handler) is handler
    assert dispatch.multi_register(["first", "second"], handler) is handler
    assert dispatch.register("decorated")(handler) is handler
    assert dispatch.multi_register(["third", "fourth"])(handler) is handler
    assert dispatch.register("keyword")(f=handler) is handler
    assert dispatch("direct", 10) == 16
    assert dispatch("first", 10) == 15
    assert dispatch("decorated", 10) == 19
    assert dispatch("third", 10) == 15
    assert dispatch("default", 10) == 3


_TYPING_CHECK = """
from typing import assert_type, overload

from probly.utils.switchdispatch import switch, switchdispatch

class Base: ...
class Derived(Base): ...

registry = switch[str, Base]()
value = Derived()
assert_type(registry.register("direct", value), Derived)
assert_type(registry.multi_register(["first", "second"], value), Derived)
assert_type(registry.register("decorated")(value), Derived)
assert_type(registry.multi_register(["third", "fourth"])(value), Derived)
assert_type(registry.register("keyword")(value=value), Derived)
assert_type(registry["direct"], Base)

from probly.quantification.decomposition.decomposition import Decomposition
from probly.quantification.notion import AleatoricUncertainty, Notion

def check_decomposition_lookup(decomposition: Decomposition) -> None:
    assert_type(decomposition[AleatoricUncertainty], AleatoricUncertainty)
    assert_type(decomposition["aleatoric"], Notion)

classes = switch[str, type[Base]]()

@classes.register("generic")
@classes.multi_register(["first", "second"])
class GenericValue[T](Base):
    def __init__(self, value: T) -> None:
        self.value = value

direct_class = classes.register("direct", GenericValue)
multiple_class = classes.multi_register(["third", "fourth"], GenericValue)
assert_type(GenericValue[float](1.0).value, float)
assert_type(direct_class[float](1.0).value, float)
assert_type(multiple_class[float](1.0).value, float)

class Subclass(GenericValue[int]): ...
class DirectSubclass(direct_class[int]): ...

@switchdispatch
def dispatch(key: str, value: int | str) -> int | str:
    return value

assert_type(dispatch("default", 1), int | str)
assert_type(dispatch(arg="default", value="text"), int | str)
dispatch("default", b"invalid")  # ty: ignore[invalid-argument-type]
dispatch("default", value=1, unexpected=True)  # ty: ignore[unknown-argument]

@overload
def handler(key: str, value: int) -> int: ...
@overload
def handler(key: str, value: str) -> str: ...
@overload
def handler(key: str, value: int | str) -> int | str: ...
def handler(key: str, value: int | str) -> int | str:
    return value

direct = dispatch.register("direct", handler)
multiple = dispatch.multi_register(["first", "second"], handler)
decorated = dispatch.register("decorated")(handler)
decorated_multiple = dispatch.multi_register(["third", "fourth"])(handler)
assert_type(direct("direct", 1), int)
assert_type(multiple("first", "a"), str)
assert_type(decorated("decorated", 1), int)
assert_type(decorated_multiple("third", "a"), str)

class Handler:
    def __call__(self, key: str, value: int | str) -> int | str:
        return value

callable_object = Handler()
assert_type(dispatch.register("object", callable_object), Handler)
assert_type(dispatch.multi_register(["object"])(callable_object), Handler)
assert_type(dispatch.register("keyword")(f=callable_object), Handler)

# Unused ignores detect accidentally accepting incompatible registrations.
registry.register("bad", 42)  # ty: ignore[invalid-argument-type]
registry.multi_register(["bad"], 42)  # ty: ignore[invalid-argument-type]
registry.register("bad")(42)  # ty: ignore[invalid-argument-type]
registry.multi_register(["bad"])(42)  # ty: ignore[invalid-argument-type]
registry.register(42, Derived())  # ty: ignore[invalid-argument-type]
registry.multi_register([42], Derived())  # ty: ignore[invalid-argument-type]

def incompatible_handler(key: str, value: bytes) -> bytes:
    return value

dispatch.register("bad", incompatible_handler)  # ty: ignore[invalid-argument-type]
dispatch.multi_register(["bad"], incompatible_handler)  # ty: ignore[invalid-argument-type]
dispatch.register("bad")(incompatible_handler)  # ty: ignore[invalid-argument-type]
dispatch.multi_register(["bad"])(incompatible_handler)  # ty: ignore[invalid-argument-type]
dispatch.register(42, callable_object)  # ty: ignore[invalid-argument-type]
"""


def test_registration_types(tmp_path: Path):
    executable = shutil.which("ty")
    if executable is None:
        pytest.skip("ty is required for the static registration checks")
    source = tmp_path / "check_registration.py"
    source.write_text(_TYPING_CHECK, encoding="utf-8")
    # Check source directly: generated partial stubs shadow imports in tests.
    source_root = Path(__file__).resolve().parents[3] / "src"
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


def test_utils_dispatchers_reject_unsupported_types() -> None:
    from probly.utils import entropy, intersection_probability  # noqa: PLC0415

    with pytest.raises(NotImplementedError, match="No entropy implementation"):
        entropy([0.5, 0.5])
    with pytest.raises(NotImplementedError, match="No intersection probability implementation"):
        intersection_probability([0.1], [0.9])
