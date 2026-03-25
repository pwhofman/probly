"""RegistryMeta is a metaclass that allows instances and subclasses to be registered in a registry."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any
from weakref import WeakSet


class RegistryMeta[T: object](ABCMeta):
    """Metaclass for registry classes."""

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

    def __all_subclasses__(cls) -> set[type]:
        """Get all subclasses, including registered ones, of cls."""
        subclasses = set(cls._subclass_registry)
        subclasses.update(cls.__subclasses__())
        return subclasses

    def __instancecheck__(cls, instance: object) -> bool:
        """Check if an instance is in the registry."""
        if instance in cls._instance_registry:
            return True

        for subclass in cls.__all_subclasses__():
            if isinstance(instance, subclass):
                return True

        return super().__instancecheck__(instance)
