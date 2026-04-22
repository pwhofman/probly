"""Tests for lazy_dispatch.singledispatch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import singledispatch as functools_singledispatch

import pytest

from flextype.registry_meta import RegistryMeta
from flextype.singledispatch import flexdispatch


def _assert_same_dispatch_result(
    ref_func: Callable[[object], object],
    got_func: Callable[[object], object],
    value: object,
) -> None:
    """Assert that lazydispatch behaves like functools.singledispatch for one input."""
    try:
        ref_result = ref_func(value)
    except Exception as ref_exc:  # noqa: BLE001
        with pytest.raises(type(ref_exc), match="Ambiguous dispatch" if isinstance(ref_exc, RuntimeError) else ""):
            got_func(value)
        return

    assert got_func(value) == ref_result


class TestFunctorParity:
    """Behavioral parity checks against functools.singledispatch."""

    def test_basic_subclass_dispatch_matches_functools(self) -> None:
        class Animal:
            pass

        class Dog(Animal):
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(Animal)
        def _ref_animal(value: Animal) -> str:
            del value
            return "animal"

        @ref.register(Dog)
        def _ref_dog(value: Dog) -> str:
            del value
            return "dog"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(Animal)
        def _got_animal(value: Animal) -> str:
            del value
            return "animal"

        @got.register(Dog)
        def _got_dog(value: Dog) -> str:
            del value
            return "dog"

        assert got(object()) == ref(object())
        assert got(Animal()) == ref(Animal())
        assert got(Dog()) == ref(Dog())

    def test_annotation_based_registration_matches_functools(self) -> None:
        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register
        def _ref_int(value: int) -> str:
            del value
            return "int"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register
        def _got_int(value: int) -> str:
            del value
            return "int"

        assert got(1) == ref(1)
        assert got("x") == ref("x")

    def test_union_registration_matches_functools(self) -> None:
        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(int | str)
        def _ref_number_or_text(value: int | str) -> str:
            del value
            return "number_or_text"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(int | str)
        def _got_number_or_text(value: int | str) -> str:
            del value
            return "number_or_text"

        assert got(1) == ref(1)
        assert got("x") == ref("x")
        assert got(1.5) == ref(1.5)

    def test_ambiguous_virtual_abc_dispatch_matches_functools(self) -> None:
        class A(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class B(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class C:
            def marker(self) -> None:
                """Concrete marker method."""

        A.register(C)
        B.register(C)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(A)
        def _ref_a(value: A) -> str:
            del value
            return "A"

        @ref.register(B)
        def _ref_b(value: B) -> str:
            del value
            return "B"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(A)
        def _got_a(value: A) -> str:
            del value
            return "A"

        @got.register(B)
        def _got_b(value: B) -> str:
            del value
            return "B"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(C())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(C())


class TestAnnotationParsing:
    """Behavior controlled by parse_annotations."""

    def test_parse_annotations_true_resolves_string_annotations_eagerly(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        @f.register
        def _int_impl(value: int) -> str:
            del value
            return "int"

        assert f(1) == "int"
        assert f("x") == "object"
        assert f.string_registry == {}

    def test_parse_annotations_false_keeps_string_annotations_lazy(self) -> None:
        @flexdispatch(parse_annotations=False)
        def f(value: object) -> str:
            del value
            return "object"

        @f.register
        def _int_impl(value: int) -> str:
            del value
            return "int"

        assert "int" in f.string_registry
        assert f(1) == "int"
        assert f.string_registry == {}

    def test_parse_annotations_true_raises_for_unresolvable_forward_ref(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        def missing_impl(value: object) -> str:
            del value
            return "missing"

        missing_impl.__annotations__["value"] = "TypeThatDoesNotExist"

        with pytest.raises(NameError):
            f.register(missing_impl)

    def test_parse_annotations_true_is_consistent_for_delayed_register(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        def missing_impl(value: object) -> None:
            del value

        missing_impl.__annotations__["value"] = "TypeThatDoesNotExist"

        with pytest.raises(NameError):
            f.delayed_register(missing_impl)


class TestLazyStringRegistration:
    """Explicit lazy registration via type strings."""

    def test_explicit_string_registration_works_for_non_builtin_type(self) -> None:
        class Base:
            pass

        class Child(Base):
            pass

        type_name = f"{Base.__module__}.{Base.__qualname__}"

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(type_name)
        def _base_impl(value: object) -> str:
            del value
            return "base"

        assert type_name in f.string_registry
        assert f(Child()) == "base"
        assert f.string_registry == {}

    def test_builtin_aliases_use_deterministic_last_registration_wins(self) -> None:
        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register("int")
        def _int_short(value: object) -> str:
            del value
            return "short"

        @f.register("builtins.int")
        def _int_full(value: object) -> str:
            del value
            return "full"

        assert list(f.string_registry.keys()) == ["int"]
        assert f(1) == "full"
        assert f(2) == "full"

    def test_builtin_aliases_reverse_order_is_also_deterministic(self) -> None:
        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register("builtins.int")
        def _int_full(value: object) -> str:
            del value
            return "full"

        @f.register("int")
        def _int_short(value: object) -> str:
            del value
            return "short"

        assert list(f.string_registry.keys()) == ["int"]
        assert f(1) == "short"
        assert f(2) == "short"


class TestDelayedRegistration:
    """Delayed registration behavior."""

    def test_delayed_registration_triggers_once_for_matching_type(self) -> None:
        calls: list[type] = []

        class Base:
            pass

        class DerivedA(Base):
            pass

        class DerivedB(Base):
            pass

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register(Base)
        def _register_base(seen_type: type) -> None:
            calls.append(seen_type)

            @f.register(Base)
            def _base_impl(value: Base) -> str:
                del value
                return "base"

        assert f(DerivedA()) == "base"
        assert f(DerivedB()) == "base"
        assert calls == [DerivedA]

    def test_delayed_registration_not_triggered_for_non_matching_type(self) -> None:
        calls: list[type] = []

        class Marker:
            pass

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register(Marker)
        def _register_marker(seen_type: type) -> None:
            calls.append(seen_type)

        assert f(1) == "object"
        assert calls == []

    def test_delayed_builtin_aliases_are_deterministic_last_wins(self) -> None:
        calls: list[str] = []

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register("builtins.int")
        def _register_full(_: type) -> None:
            calls.append("full")

            @f.register(int)
            def _full_impl(value: int) -> str:
                del value
                return "full"

        @f.delayed_register("int")
        def _register_short(_: type) -> None:
            calls.append("short")

            @f.register(int)
            def _short_impl(value: int) -> str:
                del value
                return "short"

        assert list(f.delayed_registration_registry.keys()) == ["int"]
        assert f(1) == "short"
        assert f(2) == "short"
        assert calls == ["short"]

    def test_delayed_builtin_aliases_reverse_order_is_deterministic(self) -> None:
        calls: list[str] = []

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register("int")
        def _register_short(_: type) -> None:
            calls.append("short")

            @f.register(int)
            def _short_impl(value: int) -> str:
                del value
                return "short"

        @f.delayed_register("builtins.int")
        def _register_full(_: type) -> None:
            calls.append("full")

            @f.register(int)
            def _full_impl(value: int) -> str:
                del value
                return "full"

        assert list(f.delayed_registration_registry.keys()) == ["int"]
        assert f(1) == "full"
        assert f(2) == "full"
        assert calls == ["full"]


class TestRegistryMetaDispatch:
    """Instance-level dispatch behavior for RegistryMeta registrations."""

    def test_registry_meta_instance_dispatch_basic(self) -> None:
        class RegistryType(metaclass=RegistryMeta):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        assert f(candidate) == "registry"

    def test_registry_meta_most_specific_match_wins(self) -> None:
        class RegistryBase(metaclass=RegistryMeta):
            pass

        class RegistrySub(RegistryBase):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryBase.register_instance(candidate)
        RegistrySub.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryBase)
        def _base_impl(value: RegistryBase) -> str:
            del value
            return "base"

        @f.register(RegistrySub)
        def _sub_impl(value: RegistrySub) -> str:
            del value
            return "sub"

        assert f(candidate) == "sub"

    def test_registry_meta_incomparable_matches_raise_ambiguity(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryA.register_instance(candidate)
        RegistryB.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryA)
        def _a_impl(value: RegistryA) -> str:
            del value
            return "A"

        @f.register(RegistryB)
        def _b_impl(value: RegistryB) -> str:
            del value
            return "B"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_regular_type_dispatch_more_specific_than_registry_match_wins(self) -> None:
        class RegistryType(metaclass=RegistryMeta):
            pass

        class Concrete(RegistryType):
            pass

        concrete = Concrete()

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        @f.register(Concrete)
        def _concrete_impl(value: Concrete) -> str:
            del value
            return "concrete"

        assert f(concrete) == "concrete"

    def test_same_function_with_non_matching_type_is_not_ambiguous(self) -> None:
        class Parent:
            pass

        class RegistryType(Parent, metaclass=RegistryMeta):
            pass

        class Unrelated:
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        f.register(Parent, shared)
        f.register(Unrelated, shared)

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        assert f(candidate) == "registry"

    def test_same_function_with_matching_non_parent_type_is_ambiguous(self) -> None:
        class Parent:
            pass

        class RegistryType(Parent, metaclass=RegistryMeta):
            pass

        class AlsoMatches:
            pass

        class Candidate(AlsoMatches):
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        f.register(Parent, shared)
        f.register(AlsoMatches, shared)

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_registry_meta_resolution_is_order_independent_for_equivalent_registrations(self) -> None:
        outcomes: set[str] = set()

        for order in (
            ("add_parent", "add_unrelated", "add_registry"),
            ("add_unrelated", "add_parent", "add_registry"),
            ("add_registry", "add_parent", "add_unrelated"),
            ("add_registry", "add_unrelated", "add_parent"),
        ):

            class RegistryType(metaclass=RegistryMeta):
                pass

            class Parent(RegistryType):
                pass

            class Unrelated:
                pass

            class Candidate(Parent):
                pass

            candidate = Candidate()
            RegistryType.register_instance(candidate)

            @flexdispatch
            def f(value: object) -> str:
                del value
                return "object"

            def shared(value: object) -> str:
                del value
                return "shared"

            for step in order:
                if step == "add_parent":
                    f.register(Parent, shared)
                elif step == "add_unrelated":
                    f.register(Unrelated, shared)
                else:

                    @f.register(RegistryType)
                    def _registry_impl(value: RegistryType) -> str:
                        del value
                        return "registry"

            outcomes.add(f(candidate))

        assert outcomes == {"shared"}


class TestAbcVirtualRegistrationDispatchParity:
    """Parity checks for ABC.register()-based virtual subclass dispatch."""

    def test_virtual_subclass_basic_registration_matches_functools(self) -> None:
        class VirtualABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        _assert_same_dispatch_result(ref, got, object())
        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_subclass_most_specific_match_wins_like_functools(self) -> None:
        class BaseABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class SubABC(BaseABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        BaseABC.register(VirtualImpl)
        SubABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(BaseABC)
        def _ref_base(value: BaseABC) -> str:
            del value
            return "base"

        @ref.register(SubABC)
        def _ref_sub(value: SubABC) -> str:
            del value
            return "sub"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(BaseABC)
        def _got_base(value: BaseABC) -> str:
            del value
            return "base"

        @got.register(SubABC)
        def _got_sub(value: SubABC) -> str:
            del value
            return "sub"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_subclass_incomparable_matches_raise_ambiguity_like_functools(self) -> None:
        class LeftABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class RightABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        LeftABC.register(VirtualImpl)
        RightABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(LeftABC)
        def _ref_left(value: LeftABC) -> str:
            del value
            return "left"

        @ref.register(RightABC)
        def _ref_right(value: RightABC) -> str:
            del value
            return "right"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(LeftABC)
        def _got_left(value: LeftABC) -> str:
            del value
            return "left"

        @got.register(RightABC)
        def _got_right(value: RightABC) -> str:
            del value
            return "right"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_concrete_registration_beats_virtual_abc_like_functools(self) -> None:
        class VirtualABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @ref.register(VirtualImpl)
        def _ref_concrete(value: VirtualImpl) -> str:
            del value
            return "concrete"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @got.register(VirtualImpl)
        def _got_concrete(value: VirtualImpl) -> str:
            del value
            return "concrete"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_shared_function_with_non_matching_type_still_matches_functools(self) -> None:
        class Parent:
            pass

        class Unrelated:
            pass

        class VirtualABC(Parent, ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl(Parent):
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        ref.register(Parent, shared)
        ref.register(Unrelated, shared)

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        got.register(Parent, shared)
        got.register(Unrelated, shared)

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_dispatch_is_order_independent_for_equivalent_registrations(self) -> None:
        outcomes_ref: set[str] = set()
        outcomes_got: set[str] = set()

        for order in (
            ("add_parent", "add_unrelated", "add_virtual"),
            ("add_unrelated", "add_parent", "add_virtual"),
            ("add_virtual", "add_parent", "add_unrelated"),
            ("add_virtual", "add_unrelated", "add_parent"),
        ):

            class Parent:
                pass

            class Unrelated:
                pass

            class VirtualABC(Parent, ABC):
                @abstractmethod
                def marker(self) -> None:
                    """Abstract marker method."""

            class VirtualImpl(Parent):
                def marker(self) -> None:
                    """Concrete marker method."""

            VirtualABC.register(VirtualImpl)

            @functools_singledispatch
            def ref(value: object) -> str:
                del value
                return "object"

            @flexdispatch
            def got(value: object) -> str:
                del value
                return "object"

            def shared(value: object) -> str:
                del value
                return "shared"

            for step in order:
                if step == "add_parent":
                    ref.register(Parent, shared)
                    got.register(Parent, shared)
                elif step == "add_unrelated":
                    ref.register(Unrelated, shared)
                    got.register(Unrelated, shared)
                else:

                    @ref.register(VirtualABC)
                    def _ref_virtual(value: VirtualABC) -> str:
                        del value
                        return "virtual"

                    @got.register(VirtualABC)
                    def _got_virtual(value: VirtualABC) -> str:
                        del value
                        return "virtual"

            outcomes_ref.add(ref(VirtualImpl()))
            outcomes_got.add(got(VirtualImpl()))

        assert outcomes_ref == outcomes_got == {"virtual"}


class TestSubclassHookDispatchParity:
    """Parity checks for __subclasshook__ interactions with functools.singledispatch."""

    def test_subclasshook_single_match_matches_functools(self) -> None:
        class HasFoo(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "foo")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class Foo:
            foo = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasFoo)
        def _ref_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasFoo)
        def _got_foo(value: HasFoo) -> str:
            del value
            return "foo"

        assert ref(Foo()) == "foo"
        assert got(Foo()) == ref(Foo())

    def test_subclasshook_incomparable_matches_raise_ambiguity_like_functools(self) -> None:
        class HasFoo(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "foo")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class HasBar(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "bar")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class FooBar:
            foo = 1
            bar = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasFoo)
        def _ref_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @ref.register(HasBar)
        def _ref_bar(value: HasBar) -> str:
            del value
            return "bar"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasFoo)
        def _got_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @got.register(HasBar)
        def _got_bar(value: HasBar) -> str:
            del value
            return "bar"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(FooBar())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(FooBar())

    def test_subclasshook_parent_child_overlapping_hooks_match_functools(self) -> None:
        class HasX(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "x")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class HasXY(HasX):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "x") and hasattr(subclass, "y")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class XY:
            x = 1
            y = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasX)
        def _ref_x(value: HasX) -> str:
            del value
            return "x"

        @ref.register(HasXY)
        def _ref_xy(value: HasXY) -> str:
            del value
            return "xy"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasX)
        def _got_x(value: HasX) -> str:
            del value
            return "x"

        @got.register(HasXY)
        def _got_xy(value: HasXY) -> str:
            del value
            return "xy"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(XY())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(XY())

    def test_subclasshook_notimplemented_falls_back_like_functools(self) -> None:
        class HookNotImplemented(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                del subclass
                return NotImplemented

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class Concrete:
            def marker(self) -> None:
                """Concrete marker method."""

        HookNotImplemented.register(Concrete)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HookNotImplemented)
        def _ref_impl(value: HookNotImplemented) -> str:
            del value
            return "hook"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HookNotImplemented)
        def _got_impl(value: HookNotImplemented) -> str:
            del value
            return "hook"

        assert ref(Concrete()) == "hook"
        assert got(Concrete()) == ref(Concrete())


class TestInstanceHookDispatch:
    """Behavior checks for RegistryMeta __instancehook__ interactions."""

    def test_instancehook_single_positive_match_dispatches_registry_impl(self) -> None:
        class HasFooRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "foo")

        class Candidate:
            foo = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasFooRegistry)
        def _impl(value: HasFooRegistry) -> str:
            del value
            return "foo"

        assert f(Candidate()) == "foo"

    def test_instancehook_most_specific_registry_type_wins(self) -> None:
        class HasXRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "x")

        class HasXYRegistry(HasXRegistry):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "x") and hasattr(instance, "y")

        class Candidate:
            x = 1
            y = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasXRegistry)
        def _x_impl(value: HasXRegistry) -> str:
            del value
            return "x"

        @f.register(HasXYRegistry)
        def _xy_impl(value: HasXYRegistry) -> str:
            del value
            return "xy"

        assert f(Candidate()) == "xy"

    def test_instancehook_incomparable_positive_matches_raise_ambiguity(self) -> None:
        class HasFooRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "foo")

        class HasBarRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "bar")

        class Candidate:
            foo = 1
            bar = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasFooRegistry)
        def _foo_impl(value: HasFooRegistry) -> str:
            del value
            return "foo"

        @f.register(HasBarRegistry)
        def _bar_impl(value: HasBarRegistry) -> str:
            del value
            return "bar"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(Candidate())

    def test_instancehook_notimplemented_falls_back_to_registered_instance(self) -> None:
        class HookNotImplementedRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                del instance
                return NotImplemented

        class Candidate:
            pass

        candidate = Candidate()
        HookNotImplementedRegistry.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HookNotImplementedRegistry)
        def _impl(value: HookNotImplementedRegistry) -> str:
            del value
            return "registered"

        assert f(candidate) == "registered"

    def test_instancehook_is_cached_for_hashable_instances_across_calls(self) -> None:
        class CountedRegistry(metaclass=RegistryMeta):
            calls = 0

            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                cls.calls += 1
                return hasattr(instance, "x")

        class Candidate:
            x = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(CountedRegistry)
        def _impl(value: CountedRegistry) -> str:
            del value
            return "counted"

        candidate = Candidate()
        assert f(candidate) == "counted"
        assert f(candidate) == "counted"
        assert CountedRegistry.calls == 1

    def test_instancehook_not_cached_for_unhashable_instances_across_calls(self) -> None:
        class ListRegistry(metaclass=RegistryMeta):
            calls = 0

            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                cls.calls += 1
                return isinstance(instance, list)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(ListRegistry)
        def _impl(value: ListRegistry) -> str:
            del value
            return "list"

        candidate: list[int] = []
        assert f(candidate) == "list"
        assert f(candidate) == "list"
        assert ListRegistry.calls == 2
