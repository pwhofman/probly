"""RegistryMeta is a metaclass that allows instances and subclasses to be registered in a registry."""

from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakSet

if TYPE_CHECKING:
    from collections.abc import Callable

EXCLUDED_ATTRS = frozenset(
    {
        "_subclass_registry",
        "_instance_registry",
        "_structural_checking",
    }
)


class RegistryMeta[T: object](ABCMeta):
    """Metaclass for registry classes."""

    _subclass_registry: WeakSet[type]
    _instance_registry: WeakSet[T]

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Create a new class with a registry."""
        super().__init__(name, bases, namespace, **kwargs)
        cls._subclass_registry: WeakSet[type] = WeakSet()
        cls._instance_registry: WeakSet[T] = WeakSet()

    def register(cls, subclass: type) -> type:
        """Register a subclass in the registry."""
        res = super().register(subclass)
        cls._subclass_registry.add(subclass)
        return res

    def register_instance[Q: T](cls, instance: Q) -> Q:
        """Register an instance in the registry."""
        cls._instance_registry.add(instance)
        return instance

    def _non_registered_instancecheck(cls, instance: object) -> bool:
        """Check if an instance is an instance of cls without checking the registry."""
        return super().__instancecheck__(instance)

    def __instancecheck__(cls, instance: object) -> bool:
        """Check if an instance is in the registry."""
        if instance in cls._instance_registry:
            return True

        for subclass in cls._subclass_registry:
            if isinstance(instance, subclass):
                return True
        for subclass in cls.__subclasses__():
            if isinstance(instance, subclass):
                return True

        return cls._non_registered_instancecheck(instance)


@classmethod
def _no_structural_checking_subclass_hook(cls: type, subclass: type, /) -> bool:  # noqa: ARG001
    """A __subclasshook__ that disables structural checking."""
    return NotImplemented


class ProtocolRegistryMeta[T](RegistryMeta[T], type(Protocol)):
    """Metaclass for protocol registry classes.

    Takes an additional keyword argument `structural_checking`
    which controls whether structural checking is performed, as is the default for protocols.
    If `structural_checking` is False, ProtocolRegistry classes will only consider regular inheritance and
    explicit registration, not structural compatibility, when checking for instance and subclass relationships.
    """

    _structural_checking: bool = True

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        structural_checking: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> ProtocolRegistryMeta[T]:
        """Create a new protocol registry class."""
        del structural_checking
        if Protocol not in bases and ProtocolRegistry in bases:
            bases: tuple[type, ...] = (*bases, Protocol)
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        return cls

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        structural_checking: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the protocol registry class."""
        super().__init__(name, bases, namespace, **kwargs)
        cls._structural_checking = structural_checking

        if not structural_checking:
            if "__subclasshook__" not in namespace:
                # Override Protocol's __subclasshook__ with one that disables structural checking if
                # there is no custom __subclasshook__ defined in the class body.
                cls.__subclasshook__: Callable[[type, type], bool] = _no_structural_checking_subclass_hook

            # Disable the gate that prevents Protocols from being checked via isinstance()
            # In other words: Non-structural ProtocolRegistries should be runtime_checkable by default.
            # For potential downsides of this approach, please consider cpython issue gh-113320.
            runtime_checkable(cls)

        protocol_attrs: set[str] | None = getattr(cls, "__protocol_attrs__", None)

        if protocol_attrs is not None:
            cls.__protocol_attrs__ = protocol_attrs - EXCLUDED_ATTRS

    def _non_registered_instancecheck(cls, instance: object) -> bool:
        """Check if an instance is an instance of cls without checking the registry."""
        if not cls._structural_checking:
            return ABCMeta.__instancecheck__(cls, instance)
        return super()._non_registered_instancecheck(instance)


class Registry(metaclass=RegistryMeta):
    """Helper class to create registries without needing to use the metaclass mechanism explicitly."""


class ProtocolRegistry(Protocol, metaclass=ProtocolRegistryMeta):
    """Helper class to create protocol registries without needing to use the metaclass mechanism explicitly."""
