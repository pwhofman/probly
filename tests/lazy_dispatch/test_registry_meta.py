"""Tests for the lazy_dispatch.registry_meta module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from lazy_dispatch.registry_meta import ProtocolRegistry, ProtocolRegistryMeta, RegistryMeta


class TestRegistryMeta:
    """Tests for RegistryMeta."""

    def test_registry_meta_is_metaclass_for_registry_classes(self) -> None:
        """Registry classes should themselves be instances of RegistryMeta."""

        class Base(metaclass=RegistryMeta):
            pass

        assert isinstance(Base, RegistryMeta)

    def test_register_adds_virtual_subclass_for_isinstance(self) -> None:
        """Registering a class should make its instances pass isinstance checks."""

        class Base(metaclass=RegistryMeta):
            pass

        class Virtual:
            pass

        registered = Base.register(Virtual)

        assert registered is Virtual
        assert isinstance(Virtual(), Base)

    def test_register_instance_marks_instance_for_class_and_parents(self) -> None:
        """Registered instances should match the class and its registry-meta parents."""

        class B(metaclass=RegistryMeta):
            pass

        class D(metaclass=RegistryMeta):
            pass

        class E(metaclass=RegistryMeta):
            pass

        class F(E):
            pass

        b = B()
        returned = F.register_instance(b)

        assert returned is b
        assert isinstance(b, F)
        assert isinstance(b, E)
        assert not isinstance(b, D)

    def test_instance_registration_does_not_leak_to_registered_subclasses(self) -> None:
        """Registering an instance on a class should not make it an instance of virtual subclasses."""

        class B(metaclass=RegistryMeta):
            pass

        class E(metaclass=RegistryMeta):
            pass

        class F(E):
            pass

        class G:
            pass

        F.register(G)

        b = B()
        F.register_instance(b)

        assert isinstance(b, F)
        assert not isinstance(b, G)

    def test_regular_subclass_relationships_work_without_registration(self) -> None:
        """Regular inheritance should still satisfy issubclass and isinstance."""

        class Base(metaclass=RegistryMeta):
            pass

        class Child(Base):
            pass

        child = Child()

        assert issubclass(Child, Base)
        assert isinstance(child, Child)
        assert isinstance(child, Base)


class TestProtocolRegistryMeta:
    """Tests for ProtocolRegistryMeta."""

    def test_runtime_checkable_non_method_member_matches_protocol_behavior(self) -> None:
        """ProtocolRegistry should match Protocol behavior for non-method runtime members."""

        class Candidate:
            pass

        @runtime_checkable
        class RegistryProtocol(ProtocolRegistry):
            x = 1

            def f(self) -> None:
                return None

        @runtime_checkable
        class PlainProtocol(Protocol):
            x = 1

            def f(self) -> None:
                return None

        assert isinstance(Candidate(), RegistryProtocol) is isinstance(Candidate(), PlainProtocol)

        with pytest.raises(TypeError) as registry_error:
            issubclass(Candidate, RegistryProtocol)
        with pytest.raises(TypeError) as protocol_error:
            issubclass(Candidate, PlainProtocol)

        assert str(registry_error.value) == str(protocol_error.value)

    def test_runtime_checkable_protocol_registry_uses_structural_checks(self) -> None:
        """Runtime-checkable protocols should use structural checks when enabled."""

        @runtime_checkable
        class RuntimeStructural(ProtocolRegistry):
            def f(self) -> None:
                return None

        class HasF:
            def f(self) -> None:
                return None

        class MissingF:
            pass

        assert issubclass(HasF, RuntimeStructural)
        assert isinstance(HasF(), RuntimeStructural)
        assert not issubclass(MissingF, RuntimeStructural)
        assert not isinstance(MissingF(), RuntimeStructural)

    def test_subprotocol_inheriting_protocol_registry_behaves_like_protocol(self) -> None:
        """A subprotocol should keep Protocol behavior when inheriting from ProtocolRegistry."""

        class BaseProtocol(ProtocolRegistry):
            def f(self) -> None:
                return None

        @runtime_checkable
        class SubProtocol(BaseProtocol, Protocol):
            pass

        class HasF:
            def f(self) -> None:
                return None

        class MissingF:
            pass

        assert issubclass(HasF, SubProtocol)
        assert isinstance(HasF(), SubProtocol)
        assert not issubclass(MissingF, SubProtocol)
        assert not isinstance(MissingF(), SubProtocol)

    def test_regular_subclass_of_protocol_registry_does_not_structurally_match(self) -> None:
        """A regular subclass should not become a structural protocol implicitly."""

        class BaseProtocol(ProtocolRegistry):
            def f(self) -> None:
                return None

        class RegularSubclass(BaseProtocol):
            pass

        class HasFOnly:
            def f(self) -> None:
                return None

        assert not issubclass(HasFOnly, RegularSubclass)
        assert not isinstance(HasFOnly(), RegularSubclass)

    def test_structural_checking_false_disables_protocol_runtime_structure(self) -> None:
        """`structural_checking=False` should disable Protocol-style structural matches."""

        class BaseProtocol(ProtocolRegistry, structural_checking=False):
            def f(self) -> None:
                return None

        class StructuralButNotNominal:
            def f(self, value: int) -> int:
                return value

        assert BaseProtocol._structural_checking is False  # noqa: SLF001
        assert not issubclass(StructuralButNotNominal, BaseProtocol)
        assert not isinstance(StructuralButNotNominal(), BaseProtocol)

    def test_structural_checking_false_propagates_to_protocol_subclasses(self) -> None:
        """Protocol subclasses should inherit non-structural checking from their parent."""

        class PredictorProtocol(ProtocolRegistry, Protocol, structural_checking=False):
            def predict(self) -> int:
                return 1

        class EnsemblePredictorProtocol(PredictorProtocol, Protocol):
            pass

        class RandomPredictorProtocol(PredictorProtocol, Protocol):
            pass

        class HasPredict:
            def predict(self) -> int:
                return 1

        assert PredictorProtocol._structural_checking is False  # noqa: SLF001
        assert EnsemblePredictorProtocol._structural_checking is False  # noqa: SLF001
        assert RandomPredictorProtocol._structural_checking is False  # noqa: SLF001
        assert not issubclass(HasPredict, EnsemblePredictorProtocol)
        assert not isinstance(HasPredict(), EnsemblePredictorProtocol)
        assert not issubclass(HasPredict, RandomPredictorProtocol)
        assert not isinstance(HasPredict(), RandomPredictorProtocol)

    def test_structural_checking_can_be_overridden_in_subclasses(self) -> None:
        """Subclasses should be able to override inherited non-structural checking."""

        class BaseProtocol(ProtocolRegistry, Protocol, structural_checking=False):
            def f(self) -> None:
                return None

        @runtime_checkable
        class StructuralChild(BaseProtocol, Protocol, structural_checking=True):
            pass

        class HasF:
            def f(self) -> None:
                return None

        assert BaseProtocol._structural_checking is False  # noqa: SLF001
        assert StructuralChild._structural_checking is True  # noqa: SLF001
        assert issubclass(HasF, StructuralChild)
        assert isinstance(HasF(), StructuralChild)

    def test_non_protocol_subclass_allows_structural_checking_false(self) -> None:
        """Non-protocol subclasses should accept structural_checking=False without type errors."""

        class BaseProtocol(ProtocolRegistry, Protocol, structural_checking=False):
            def f(self) -> None:
                return None

        class ConcreteChild(BaseProtocol, structural_checking=False):
            pass

        class HasF:
            def f(self) -> None:
                return None

        child = ConcreteChild()

        assert ConcreteChild._structural_checking is False  # noqa: SLF001
        assert issubclass(ConcreteChild, BaseProtocol)
        assert isinstance(child, BaseProtocol)
        assert not issubclass(HasF, ConcreteChild)
        assert not isinstance(HasF(), ConcreteChild)

    def test_non_protocol_subclass_custom_hook_is_respected_in_nominal_mode(self) -> None:
        """Custom hooks on non-protocol subclasses should still drive subclass decisions."""

        class BaseProtocol(ProtocolRegistry, Protocol, structural_checking=False):
            def f(self) -> None:
                return None

        class ConcreteChild(BaseProtocol, structural_checking=False):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return True

        class Candidate:
            pass

        assert issubclass(Candidate, ConcreteChild)
        assert isinstance(Candidate(), ConcreteChild)

    def test_direct_metaclass_use_honors_structural_checking_false(self) -> None:
        """Direct use of the metaclass should also disable Protocol structural matching."""

        class DirectProtocol(Protocol, metaclass=ProtocolRegistryMeta, structural_checking=False):
            def f(self) -> None:
                return None

        class StructuralButNotNominal:
            def f(self, value: int) -> int:
                return value

        assert DirectProtocol._structural_checking is False  # noqa: SLF001
        assert not issubclass(StructuralButNotNominal, DirectProtocol)
        assert not isinstance(StructuralButNotNominal(), DirectProtocol)

    def test_structural_checking_false_with_non_method_member_uses_nominal_checks(self) -> None:
        """Non-method members should not force Protocol TypeErrors when structural checks are disabled."""

        class BaseProtocol(ProtocolRegistry, structural_checking=False):
            x = 1

            def f(self) -> None:
                return None

        class Candidate:
            pass

        assert not isinstance(Candidate(), BaseProtocol)
        assert not issubclass(Candidate, BaseProtocol)

    def test_structural_checking_false_with_custom_hook_allows_runtime_checks(self) -> None:
        """Custom subclass hooks should not block runtime checks in non-structural mode."""

        class BaseProtocol(ProtocolRegistry, structural_checking=False):
            def f(self) -> None:
                return None

            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return False

        class Candidate:
            pass

        assert not issubclass(Candidate, BaseProtocol)
        assert not isinstance(Candidate(), BaseProtocol)

    def test_protocol_registry_register_and_register_instance(self) -> None:
        """ProtocolRegistry should support register and register_instance with pre/post checks."""

        class BaseProtocol(ProtocolRegistry, structural_checking=False):
            def f(self) -> None:
                return None

        class Virtual:
            pass

        class Concrete:
            pass

        virtual_instance = Virtual()
        concrete_instance = Concrete()
        another_concrete_instance = Concrete()

        assert not issubclass(Virtual, BaseProtocol)
        assert not isinstance(virtual_instance, BaseProtocol)
        assert not isinstance(concrete_instance, BaseProtocol)
        assert not isinstance(another_concrete_instance, BaseProtocol)

        registered = BaseProtocol.register(Virtual)
        assert registered is Virtual
        assert issubclass(Virtual, BaseProtocol)
        assert isinstance(virtual_instance, BaseProtocol)

        returned_instance = BaseProtocol.register_instance(concrete_instance)
        assert returned_instance is concrete_instance
        assert isinstance(concrete_instance, BaseProtocol)
        assert not isinstance(another_concrete_instance, BaseProtocol)

    def test_protocol_registry_meta_register_and_register_instance(self) -> None:
        """Direct ProtocolRegistryMeta use should support register and register_instance."""

        class DirectProtocol(Protocol, metaclass=ProtocolRegistryMeta, structural_checking=False):
            def f(self) -> None:
                return None

        class Virtual:
            pass

        class Concrete:
            pass

        virtual_instance = Virtual()
        concrete_instance = Concrete()
        another_concrete_instance = Concrete()

        assert not issubclass(Virtual, DirectProtocol)
        assert not isinstance(virtual_instance, DirectProtocol)
        assert not isinstance(concrete_instance, DirectProtocol)
        assert not isinstance(another_concrete_instance, DirectProtocol)

        registered = DirectProtocol.register(Virtual)
        assert registered is Virtual
        assert issubclass(Virtual, DirectProtocol)
        assert isinstance(virtual_instance, DirectProtocol)

        returned_instance = DirectProtocol.register_instance(concrete_instance)
        assert returned_instance is concrete_instance
        assert isinstance(concrete_instance, DirectProtocol)
        assert not isinstance(another_concrete_instance, DirectProtocol)

    def test_regular_subclass_relationships_work_for_protocol_registry_meta(self) -> None:
        """Regular inheritance should still satisfy issubclass and isinstance."""

        class DirectProtocol(Protocol, metaclass=ProtocolRegistryMeta, structural_checking=False):
            def f(self) -> None:
                return None

        class Child(DirectProtocol):
            pass

        child = Child()

        assert issubclass(Child, DirectProtocol)
        assert isinstance(child, Child)
        assert isinstance(child, DirectProtocol)
