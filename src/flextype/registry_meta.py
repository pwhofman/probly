"""RegistryMeta is a metaclass that allows instances and subclasses to be registered in a registry."""

from __future__ import annotations

from abc import ABCMeta
import functools
from typing import TYPE_CHECKING, Any, Protocol, is_protocol, runtime_checkable
from weakref import WeakSet

from flextype.isinstance import _find_closest_string_type, _split_lazy_type

if TYPE_CHECKING:
    from collections.abc import Callable

    from flextype import LazyType

EXCLUDED_ATTRS = frozenset(
    {
        "_subclass_registry",
        "_instance_registry",
        "_negative_instance_registry",
        "_string_registry",
        "_structural_checking",
    }
)


def iter_registry_classes() -> list[RegistryMeta[Any]]:
    """Return all currently alive classes that use RegistryMeta."""
    classes = [
        registry_class
        for registry_class in RegistryMeta.known_registry_classes
        if isinstance(registry_class, RegistryMeta)
    ]
    classes.sort(key=lambda cls: (cls.__module__, cls.__qualname__))
    return classes


class RegistrationError(Exception):
    """Exception raised when an object cannot be registered."""

    def __init__(self, *, registry: type, target: object) -> None:
        """Create a registration error with contextual metadata."""
        self.registry = registry
        self.target_type = type(target)
        super().__init__(
            f"Registration failed for registry class {registry.__qualname__!r} and target type "
            f"{self.target_type.__qualname__!r}: "
            "Registered instances must be weak-referenceable and hashable."
        )


@classmethod
def _lazy_subclass_hook[T](cls: RegistryMeta[T], subclass: type, /) -> bool:
    """A __subclasshook__ that checks whether the ."""
    string_registry = getattr(cls, "_string_registry", None)

    if string_registry is not None and len(string_registry) > 0:
        closest = _find_closest_string_type(subclass, string_registry)
        if closest is not None:
            real_type, string_type = closest
            string_registry.remove(string_type)
            cls._register(real_type)

    return NotImplemented


@classmethod
def _nop_instancehook[T](_cls: RegistryMeta[T], _instance: object, /) -> bool:
    """A __instancehook__ that does nothing."""
    return NotImplemented


def _lazy_subclass_hook_with_pre_hook[T](
    pre_hook: Callable[[RegistryMeta[T], type], bool],
) -> Callable[[RegistryMeta[T], type], bool]:
    """A __subclasshook__ that checks the pre_hook before checking the string registry."""

    @classmethod
    def hook(cls: RegistryMeta[T], subclass: type, /) -> bool:
        pre_res = pre_hook(cls, subclass)

        if pre_res is not NotImplemented:
            return pre_res

        return _lazy_subclass_hook.__func__(cls, subclass)

    return hook


class RegistryMeta[T: object](ABCMeta):
    """Metaclass for registry classes."""

    _subclass_registry: WeakSet[type]
    _instance_registry: WeakSet[T]
    _negative_instance_registry: WeakSet[object]
    _string_registry: set[str]
    known_registry_classes: WeakSet[type] = WeakSet()

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
        cls._negative_instance_registry: WeakSet[object] = WeakSet()
        cls._string_registry: set[str] = set()

        if not is_protocol(cls):
            subclasshook = namespace.get("__subclasshook__")
            if subclasshook is None:
                subclasshook = _lazy_subclass_hook
            else:
                if isinstance(subclasshook, classmethod):
                    subclasshook = subclasshook.__func__
                subclasshook = _lazy_subclass_hook_with_pre_hook(subclasshook)

            cls.__subclasshook__ = subclasshook  # ty: ignore[invalid-assignment]

        RegistryMeta.known_registry_classes.add(cls)

    def is_explicit_instance_registered(cls, instance: object) -> bool:
        """Return whether `instance` was explicitly registered via register_instance."""
        try:
            return instance in cls._instance_registry
        except TypeError:
            return False

    def _register(cls, subclass: type) -> type:
        """Register a subclass in the registry."""
        res = super().register(subclass)
        cls._subclass_registry.add(subclass)
        return res

    def _register_lazy(cls, subclass_strings: set[str]) -> None:
        if is_protocol(cls) and len(subclass_strings) > 0:
            msg = (
                "Lazy subclass registration not supported for Protocols. Use ProtocolRegistry "
                "with structural_checking=False instead if you want to use lazy subclass registration."
            )
            raise RuntimeError(msg)
        cls._string_registry.update(subclass_strings)

    def register(cls, subclass: LazyType) -> type:
        """Register a (lazy) subclass or a set of subclasses in registry."""
        types, strings = _split_lazy_type(subclass)
        for t in types:
            cls._register(t)

        cls._register_lazy(strings)
        cls._negative_instance_registry = WeakSet()

        if isinstance(subclass, type):
            return subclass
        return cls

    def _register_instance[Q](cls: RegistryMeta[T], instance: Q) -> Q:
        try:
            cls._instance_registry.add(instance)  # ty:ignore[invalid-argument-type]
            cls._negative_instance_registry.discard(instance)
        except TypeError as err:
            raise RegistrationError(
                registry=cls,
                target=instance,
            ) from err
        return instance

    def _negative_register_instance[Q](cls: RegistryMeta[T], instance: Q) -> Q:
        try:
            cls._negative_instance_registry.add(instance)
            cls._instance_registry.discard(instance)  # ty:ignore[invalid-argument-type]
        except TypeError as err:
            raise RegistrationError(
                registry=cls,
                target=instance,
            ) from err
        return instance

    def register_instance[Q](cls: RegistryMeta[T], instance: Q) -> Q:
        """Register an instance in the registry."""
        if isinstance(instance, cls):
            return instance

        return cls._register_instance(instance)

    def register_factory[**In, Q](cls: RegistryMeta[T], func: Callable[In, Q]) -> Callable[In, Q]:
        """Decorator to annotate the results of a function with the registry type."""
        return annotator(cls)(func)

    def _non_registered_instancecheck(cls, instance: object) -> bool:
        """Check if an instance is an instance of cls without checking the registry."""
        return super().__instancecheck__(instance)

    def __instancecheck__(cls, instance: object) -> bool:
        """Check if an instance is in the registry."""
        try:
            if instance in cls._instance_registry:
                return True
            if instance in cls._negative_instance_registry:
                return False
        except TypeError:
            pass

        instancehook = getattr(cls, "__instancehook__", None)
        if instancehook is not None:
            res = instancehook(instance)
            if res is not NotImplemented:
                try:
                    if res:
                        cls._register_instance(instance)
                    else:
                        cls._negative_register_instance(instance)
                except RegistrationError:
                    pass

                return res

        for subclass in cls._subclass_registry:
            if isinstance(instance, subclass):
                return True
        for subclass in cls.__subclasses__():
            if isinstance(instance, subclass):
                return True

        return cls._non_registered_instancecheck(instance)


class ProtocolRegistryMeta[T](RegistryMeta[T], type(Protocol)):
    """Metaclass for protocol registry classes.

    Takes an additional keyword argument `structural_checking`
    which controls whether structural checking is performed, as is the default for protocols.
    If `structural_checking` is False, ProtocolRegistry classes will only consider regular inheritance and
    explicit registration, not structural compatibility, when checking for instance and subclass relationships.
    `_structural_checking` is True by default, so that ProtocolRegistryMeta behaves like Protocol by default.
    The chosen checking behavior is inherited by subclasses, but can be overridden by passing `structural_checking`
    to the class definition.
    """

    _structural_checking: bool = True

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        structural_checking: bool | None = None,
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
        structural_checking: bool | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the protocol registry class."""
        super().__init__(name, bases, namespace, **kwargs)
        if structural_checking is not None:
            cls._structural_checking = structural_checking
        else:
            structural_checking = cls._structural_checking

        if not structural_checking:
            # Override Protocol's __subclasshook__ with one that disables structural checking
            # and instead adds lazy subclass registration support to the hook.

            # This has to be done here, because Protocol inspects the callstack of __subclasscheck__ to determine
            # whether the subclass check is being called from an isinstance check coming from ABCMeta;
            # see __allow_reckless_class_checks and _ProtocolMeta in typing for details.
            # To align the behavior of ProtocolRegistry and Protocol, __subclasshook__ has to be used.
            subclasshook: Callable[[RegistryMeta[T], type], bool] | classmethod | None = namespace.get(
                "__subclasshook__"
            )
            if subclasshook is None:
                subclasshook = _lazy_subclass_hook
            else:
                if isinstance(subclasshook, classmethod):
                    subclasshook = subclasshook.__func__
                subclasshook = _lazy_subclass_hook_with_pre_hook(subclasshook)
            cls.__subclasshook__ = subclasshook  # ty:ignore[invalid-assignment]

            if is_protocol(cls):
                # Disable the gate that prevents Protocols from being checked via isinstance()
                # In other words: Non-structural ProtocolRegistries should be runtime_checkable by default.
                # For potential downsides of this approach, please consider cpython issue gh-113320.
                runtime_checkable(cls)

        protocol_attrs: set[str] | None = getattr(cls, "__protocol_attrs__", None)

        if protocol_attrs is not None:
            cls.__protocol_attrs__ = protocol_attrs - EXCLUDED_ATTRS

        if hasattr(cls, "__instancehook__") and "__instancehook__" not in cls.__dict__:
            # For Protocols, __instancehook__ should not be inherited to derived ProtocolRegistries.
            # This mirrors the behavior of __subclasshook__ in Protocols.
            cls.__instancehook__ = _nop_instancehook

    def _register_lazy(cls, subclass_strings: set[str]) -> None:
        if len(subclass_strings) > 0 and cls._structural_checking:
            msg = "Lazy subclass registration not supported for ProtocolRegistry with structural_checking=True."
            raise RuntimeError(msg)
        cls._string_registry.update(subclass_strings)

    def _non_registered_instancecheck(cls, instance: object) -> bool:
        """Check if an instance is an instance of cls without checking the registry."""
        if not cls._structural_checking:
            return ABCMeta.__instancecheck__(cls, instance)
        return super()._non_registered_instancecheck(instance)


class Registry[T](metaclass=RegistryMeta):
    """Helper class to create registries without needing to use the metaclass mechanism explicitly."""


class ProtocolRegistry[T](Protocol, metaclass=ProtocolRegistryMeta):
    """Helper class to create protocol registries without needing to use the metaclass mechanism explicitly."""


class _RegistryAnnotator[T: object](Protocol):
    """Callable protocol for decorators that preserve input signature and change return type."""

    def __call__[**In, Q](self, func: Callable[In, Q], /) -> Callable[In, Q]:
        """Decorate `func` while preserving its parameters."""


def annotator[T: object](registry_type: RegistryMeta[T]) -> _RegistryAnnotator[T]:
    """Decorator to annotate the result of a function with a registry type.

    This is useful for functions that return instances of a registry, but where the return type is not known statically,
    e.g. because the function is a lazy dispatch function that can return different types.
    """

    def decorator[**In, Q](func: Callable[In, Q]) -> Callable[In, Q]:
        @functools.wraps(func)
        def wrapper(*args: In.args, **kwargs: In.kwargs) -> Q:
            res = func(*args, **kwargs)
            try:
                return registry_type.register_instance(res)
            except RegistrationError:
                return res  # If the result cannot be registered, return it as is.

        return wrapper

    return decorator
