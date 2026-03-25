"""Tests for the lazy_dispatch.registry_meta module."""

from __future__ import annotations

from lazy_dispatch.registry_meta import RegistryMeta


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

    def test_all_subclasses_includes_real_and_registered_subclasses(self) -> None:
        """Real subclasses and registered virtual subclasses should both be returned."""

        class Base(metaclass=RegistryMeta):
            pass

        class RealChild(Base):
            pass

        class VirtualChild:
            pass

        Base.register(VirtualChild)

        subclasses = Base.__all_subclasses__()

        assert RealChild in subclasses
        assert VirtualChild in subclasses

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
