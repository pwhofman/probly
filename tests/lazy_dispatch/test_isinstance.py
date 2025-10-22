"""Tests for the probly.lazy_dispatch.isinstance module."""

from __future__ import annotations

import pytest

from lazy_dispatch.isinstance import LazyType, lazy_isinstance


class TestIsInstance:
    """Tests for the lazy_isinstance function."""

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, int, True),
            (5, (int, str), True),
            (5, int | str, True),
            ("hello", str, True),
            ("hello", (int, str), True),
            ("hello", int | str, True),
            (5.0, int, False),
            (5.0, (int, str), False),
            (5.0, int | str, False),
            ([1, 2, 3], list, True),
            ([1, 2, 3], (list, dict), True),
            ([1, 2, 3], list | dict, True),
            ({"a": 1}, dict, True),
            ({"a": 1}, (list, dict), True),
            ({"a": 1}, list | (dict | list), True),
            (None, type(None), True),
            (None, (int, type(None)), True),
            (None, int | type(None), True),
        ],
    )
    def test_eager_builtin_types(self, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test lazy_isinstance with real types."""
        assert lazy_isinstance(obj, classinfo) == expected

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, "int", True),
            (5, ("builtins.int", "builtins.str"), True),
            ("hello", "str", True),
            ("hello", ("int", "str"), True),
            (5.0, "builtins.int", False),
            (5.0, ("int", "str"), False),
            ([1, 2, 3], "builtins.list", True),
            ([1, 2, 3], ("list", "dict"), True),
            ({"a": 1}, "dict", True),
            ({"a": 1}, ("list", "builtins.dict"), True),
        ],
    )
    def test_lazy_builtin_types(self, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test lazy_isinstance with stringified types."""
        assert lazy_isinstance(obj, classinfo) == expected

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, (str, "builtins.int"), True),
            (5, (int, "str"), True),
            ("hello", (int, "str"), True),
            ("hello", (str, "int"), True),
            (5.0, (str, "builtins.int"), False),
            (5.0, ("str", int), False),
            ([1, 2, 3], (dict, "builtins.list"), True),
            ([1, 2, 3], (list, "dict"), True),
            ({"a": 1}, (dict, "builtins.dict"), True),
            ({"a": 1}, ("dict", list), True),
        ],
    )
    def test_mixed_builtin_types(self, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test lazy_isinstance with mixtures of real and stringified types."""
        assert lazy_isinstance(obj, classinfo) == expected
