"""Tests for the affine dispatch are redundant, because only differs in the names."""

from __future__ import annotations

from typing import cast

import pytest

from probly.calibration.scaling import common


class DummyTensor:
    """Minimal Tensor replacement."""


class DummyModule:
    """Minimal Module replacement."""

    def __call__(self, x: DummyTensor) -> DummyTensor:
        return self.forward(x)

    def forward(self, x: DummyTensor) -> DummyTensor:
        return x


class DummyTemperature:
    def __init__(self, base: DummyModule) -> None:
        """Constructor for the dummy temperature wrapper."""
        self.base = base

    def forward(self, x: DummyTensor) -> DummyTensor:
        return self.base(x)


@pytest.fixture
def dummy_model() -> DummyModule:
    return DummyModule()


@pytest.fixture(autouse=True)
def register_dummy_temperature(dummy_model: DummyModule) -> None:
    """Register DummyTemperature for dummy_model."""

    @common.register_temperature_factory(type(dummy_model))
    def _register(_base: DummyModule) -> type[DummyTemperature]:
        return DummyTemperature


def test_temperature_dispatch_returns_instance(dummy_model: DummyModule) -> None:
    """Temperature.__new__ should return instance of registered class."""
    # Cast as DummyTemperature so the type is known for static type checking (MyPy).
    temperature_instance = cast(DummyTemperature, common.temperature(dummy_model))

    assert isinstance(temperature_instance, DummyTemperature)
    assert temperature_instance.base is dummy_model


def test_temperature_dispatch_multiple_instances(dummy_model: DummyModule) -> None:
    """Dispatcher should handle different instances correctly and not overwrite anything."""
    # Cast as DummyTemperature so the type is known for static type checking (MyPy).
    instance1 = cast(DummyTemperature, common.temperature(dummy_model))
    instance2 = cast(DummyTemperature, common.temperature(dummy_model))

    assert instance1 is not instance2
    assert instance1.base is dummy_model
    assert instance2.base is dummy_model


def test_temperature_dispatch_raises_for_unregistered_model() -> None:
    """Dispatcher should throw error a NotImplementedError, when using an unregistered Model."""

    class UnregisteredModel:
        def forward(self, x: DummyTensor) -> DummyTensor:
            return x

    model = UnregisteredModel()

    with pytest.raises(NotImplementedError):
        common.temperature(model)
