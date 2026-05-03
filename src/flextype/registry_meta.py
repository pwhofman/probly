"""RegistryMeta is a metaclass that allows instances and subclasses to be registered in a registry."""

from __future__ import annotations

from abc import ABCMeta
import functools
from typing import TYPE_CHECKING, Any, Protocol, is_protocol, overload, runtime_checkable
from weakref import ReferenceType, WeakSet, ref

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


def _iter_registry_classes() -> list[RegistryMeta[Any]]:
    """Return all currently alive classes that use RegistryMeta."""
    classes = [
        registry_class
        for registry_class in RegistryMeta.known_registry_classes
        if isinstance(registry_class, RegistryMeta)
    ]
    classes.sort(key=lambda cls: (cls.__module__, cls.__qualname__))
    return classes


def get_explicit_registry_classes(instance: object) -> list[RegistryMeta[Any]]:
    """Collect all registry classes where `instance` was explicitly registered."""
    return [
        registry_class
        for registry_class in _iter_registry_classes()
        if registry_class.is_explicit_instance_registered(instance)
    ]


def copy_explicit_registry_classes(source: object, target: object) -> None:
    """Copy explicit registry registrations from `source` to `target`."""
    for registry_class in get_explicit_registry_classes(source):
        registry_class.register_instance(target)


class RegistrationError(Exception):
    """Exception raised when an object cannot be registered."""

    def __init__(self, *, registry: type, target: object) -> None:
        """Create a registration error with contextual metadata."""
        self.registry = registry
        self.target_type = type(target)
        super().__init__(
            f"Registration failed for registry class {registry.__qualname__!r} and target type "
            f"{self.target_type.__qualname__!r}: Registered instances must be weak-referenceable."
        )


class _IdentityWeakSet[T: object]:
    """Weak set that tracks objects by identity rather than equality/hash."""

    __slots__ = (
        "__weakref__",
        "_refs",
    )
    _refs: dict[int, ReferenceType[T]]

    def __init__(self) -> None:
        self._refs = {}

    def add(self, instance: T) -> None:
        """Add `instance` without requiring it to be hashable."""
        key = id(instance)
        registry_ref = ref(self)

        def remove(instance_ref: ReferenceType[T], /, *, key: int = key) -> None:
            registry = registry_ref()
            if registry is not None:
                registry.discard_ref(key, instance_ref)

        self._refs[key] = ref(instance, remove)

    def discard_ref(self, key: int, instance_ref: ReferenceType[T]) -> None:
        """Discard a weak reference if it is still the one registered for `key`."""
        if self._refs.get(key) is instance_ref:
            self._refs.pop(key, None)

    def discard(self, instance: object) -> None:
        """Discard `instance` if this exact object is registered."""
        key = id(instance)
        instance_ref = self._refs.get(key)
        if instance_ref is not None and instance_ref() is instance:
            self._refs.pop(key, None)

    def __contains__(self, instance: object) -> bool:
        """Return whether this exact object is registered."""
        key = id(instance)
        instance_ref = self._refs.get(key)
        if instance_ref is None:
            return False

        registered_instance = instance_ref()
        if registered_instance is instance:
            return True
        if registered_instance is None:
            self._refs.pop(key, None)
        return False


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


class List(list):
    """A wrapper around list that allows weak references, to allow lists to be registered in registries."""


class Dict(dict):
    """A wrapper around dict that allows weak references, to allow dicts to be registered in registries."""


class Set(set):
    """A wrapper around set that allows weak references, to allow sets to be registered in registries."""


def make_builtin_weakrefable[T: object](obj: T) -> T:
    """Wrap built-in types in a weak-referenceable wrapper to allow them to be registered in registries."""
    if type(obj) is list:
        return List(obj)  # ty:ignore[invalid-return-type]
    if type(obj) is dict:
        return Dict(obj)  # ty:ignore[invalid-return-type]
    if type(obj) is set:
        return Set(obj)  # ty:ignore[invalid-return-type]
    return obj


class RegistryMeta[T: object](ABCMeta):
    """Metaclass for registry classes."""

    _subclass_registry: WeakSet[type]
    _instance_registry: _IdentityWeakSet[T]
    _negative_instance_registry: _IdentityWeakSet[object]
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
        cls._instance_registry: _IdentityWeakSet[T] = _IdentityWeakSet()
        cls._negative_instance_registry: _IdentityWeakSet[object] = _IdentityWeakSet()
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
        cls._negative_instance_registry = _IdentityWeakSet()

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
            cls._instance_registry.discard(instance)
        except TypeError as err:
            raise RegistrationError(
                registry=cls,
                target=instance,
            ) from err
        return instance

    def register_instance[Q](cls: RegistryMeta[T], instance: Q, autocast_builtins: bool = False) -> Q:
        """Register an instance in the registry."""
        if isinstance(instance, cls):
            return instance

        if autocast_builtins:
            instance = make_builtin_weakrefable(instance)

        return cls._register_instance(instance)

    @overload
    def register_factory[**In, Q](
        cls: RegistryMeta[T],
        func: Callable[In, Q],
        *,
        autocast_builtins: bool = False,
        raise_on_failure: bool = True,
    ) -> Callable[In, Q]: ...

    @overload
    def register_factory[**In, Q](
        cls: RegistryMeta[T],
        *,
        autocast_builtins: bool = False,
        raise_on_failure: bool = True,
    ) -> Callable[[Callable[In, Q]], Callable[In, Q]]: ...

    def register_factory[**In, Q](
        cls: RegistryMeta[T],
        func: Callable[In, Q] | None = None,
        *,
        autocast_builtins: bool = False,
        raise_on_failure: bool = True,
    ) -> Callable[In, Q] | Callable[[Callable[In, Q]], Callable[In, Q]]:
        """Decorator to annotate the results of a function with the registry type."""
        if func is None:

            def decorator(func: Callable[In, Q]) -> Callable[In, Q]:
                return cls.register_factory(
                    func, autocast_builtins=autocast_builtins, raise_on_failure=raise_on_failure
                )

            return decorator

        return annotator(cls, autocast_builtins=autocast_builtins, raise_on_failure=raise_on_failure)(func)

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


def annotator[T: object](
    registry_type: RegistryMeta[T],
    autocast_builtins: bool = False,
    raise_on_failure: bool = True,
) -> _RegistryAnnotator[T]:
    """Decorator to annotate the result of a function with a registry type.

    This is useful for functions that return instances of a registry, but where the return type is not known statically,
    e.g. because the function is a lazy dispatch function that can return different types.
    """

    def decorator[**In, Q](func: Callable[In, Q]) -> Callable[In, Q]:
        @functools.wraps(func)
        def wrapper(*args: In.args, **kwargs: In.kwargs) -> Q:
            res = func(*args, **kwargs)
            try:
                return registry_type.register_instance(res, autocast_builtins=autocast_builtins)
            except RegistrationError:
                if raise_on_failure:
                    raise
                return res  # If the result cannot be registered, return it as is.

        return wrapper

    return decorator
