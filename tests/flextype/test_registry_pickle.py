"""Tests for lazy_dispatch.registry_pickle."""

from __future__ import annotations

import pickle

from flextype import registry_pickle
from flextype.registry_meta import RegistryMeta


class _RegistryAlpha(metaclass=RegistryMeta):
    pass


class _RegistryBeta(metaclass=RegistryMeta):
    pass


class _RegistryGamma(metaclass=RegistryMeta):
    pass


class _ExternalPayload:
    def __init__(self, value: int) -> None:
        self.value = value


class _SlotsPayload:
    __slots__ = ("__weakref__", "value")

    def __init__(self, value: int) -> None:
        self.value = value


class _StateRecorder:
    def __init__(self, value: int) -> None:
        self.value = value
        self.last_seen_state: dict[str, object] | None = None

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.last_seen_state = dict(state)


class _NominalBase(metaclass=RegistryMeta):
    pass


class _NominalChild(_NominalBase):
    def __init__(self, value: int) -> None:
        self.value = value


class _VirtualForNominalBase:
    def __init__(self, value: int) -> None:
        self.value = value


def test_custom_pickle_roundtrip_restores_explicit_instance_registrations() -> None:
    """Explicit register_instance relationships should survive registry pickle round-trips."""
    instance = _ExternalPayload(3)
    _RegistryAlpha.register_instance(instance)
    _RegistryBeta.register_instance(instance)

    restored = registry_pickle.loads(registry_pickle.dumps(instance))

    assert restored.value == 3
    assert isinstance(restored, _RegistryAlpha)
    assert isinstance(restored, _RegistryBeta)
    assert not isinstance(restored, _RegistryGamma)


def test_roundtrip_without_explicit_registration_behaves_like_regular_pickle() -> None:
    """Objects without explicit registry membership should be restored unchanged."""
    instance = _ExternalPayload(5)

    restored = registry_pickle.loads(registry_pickle.dumps(instance))

    assert restored.value == 5
    assert not isinstance(restored, _RegistryAlpha)


def test_roundtrip_does_not_convert_nominal_or_virtual_matches_to_explicit_memberships() -> None:
    """Nominal inheritance and class registration should not be restored as explicit instance registrations."""
    _NominalBase.register(_VirtualForNominalBase)

    child = _NominalChild(7)
    virtual = _VirtualForNominalBase(9)

    restored_child = registry_pickle.loads(registry_pickle.dumps(child))
    restored_virtual = registry_pickle.loads(registry_pickle.dumps(virtual))

    assert isinstance(restored_child, _NominalBase)
    assert isinstance(restored_virtual, _NominalBase)
    assert not _NominalBase.is_explicit_instance_registered(restored_child)
    assert not _NominalBase.is_explicit_instance_registered(restored_virtual)


def test_custom_setstate_receives_original_state_shape() -> None:
    """Custom __setstate__ should receive the original state without registry metadata wrappers."""
    instance = _StateRecorder(11)
    _RegistryGamma.register_instance(instance)

    restored = registry_pickle.loads(registry_pickle.dumps(instance))

    assert isinstance(restored, _RegistryGamma)
    assert restored.last_seen_state == {"value": 11, "last_seen_state": None}


def test_registry_pickle_stream_is_compatible_with_stdlib_unpickler() -> None:
    """registry_pickle.dumps output should load correctly via pickle.loads."""
    instance = _ExternalPayload(13)
    _RegistryAlpha.register_instance(instance)

    restored = pickle.loads(registry_pickle.dumps(instance))  # noqa: S301

    assert restored.value == 13
    assert isinstance(restored, _RegistryAlpha)


def test_stdlib_pickle_roundtrip_does_not_restore_explicit_instance_registration() -> None:
    """Regular pickle should not preserve explicit registry instance registrations."""
    instance = _ExternalPayload(21)
    _RegistryAlpha.register_instance(instance)

    restored = pickle.loads(pickle.dumps(instance))  # noqa: S301

    assert restored.value == 21
    assert not isinstance(restored, _RegistryAlpha)


def test_slots_only_non_registry_class_roundtrip_restores_registration() -> None:
    """Slots-only non-registry classes should also keep explicit registrations."""
    instance = _SlotsPayload(34)
    _RegistryBeta.register_instance(instance)

    restored = registry_pickle.loads(registry_pickle.dumps(instance))

    assert restored.value == 34
    assert isinstance(restored, _RegistryBeta)
