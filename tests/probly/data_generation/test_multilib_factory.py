"""Tests for data gen factory.

tests validate that create_data_generator returns correct
framework-specific generator without importing heavy ML frameworks.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, cast

import pytest


# stubs for type-checking clarity
class _BaseStub:
    def __init__(self, *, model: object, dataset: object, batch_size: int = 32, device: str | None = None) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device


class JAXStub(_BaseStub):
    pass


class TorchStub(_BaseStub):
    pass


class TFStub(_BaseStub):
    pass


def _prepare_factory_with_stubs() -> tuple[types.ModuleType, type, type, type]:
    """Inject stub generator modules and import the factory module.

    Returns a tuple of (factory_module, JAXStub, TorchStub, TFStub)
    """
    # Build stub modules using ModuleType and attribute assignment
    jax_mod = cast(Any, types.ModuleType("probly.data_generation.jax_generator"))
    torch_mod = cast(Any, types.ModuleType("probly.data_generation.pytorch_generator"))
    tf_mod = cast(Any, types.ModuleType("probly.data_generation.tensorflow_generator"))

    jax_mod.JAXDataGenerator = JAXStub
    torch_mod.PyTorchDataGenerator = TorchStub
    tf_mod.TensorFlowDataGenerator = TFStub

    sys.modules["probly.data_generation.jax_generator"] = jax_mod
    sys.modules["probly.data_generation.pytorch_generator"] = torch_mod
    sys.modules["probly.data_generation.tensorflow_generator"] = tf_mod

    # Ensure clean import of factory
    sys.modules.pop("probly.data_generation.factory", None)

    factory = importlib.import_module("probly.data_generation.factory")
    return factory, JAXStub, TorchStub, TFStub


def test_unknown_framework_raises_value_error() -> None:
    factory, *_ = _prepare_factory_with_stubs()
    with pytest.raises(ValueError, match=r"Unknown framework: unknown"):
        factory.create_data_generator(framework="unknown", model=None, dataset=None)


def test_pytorch_factory_returns_torch_generator() -> None:
    factory, _, torch_stub, _ = _prepare_factory_with_stubs()

    gen = factory.create_data_generator(
        framework="pytorch",
        model="torch_model",
        dataset="torch_dataset",
        batch_size=16,
        device="cpu",
    )

    assert isinstance(gen, torch_stub)
    gen_stub = cast(_BaseStub, gen)
    assert gen_stub.model == "torch_model"
    assert gen_stub.dataset == "torch_dataset"
    assert gen_stub.batch_size == 16
    assert gen_stub.device == "cpu"


def test_jax_factory_returns_jax_generator() -> None:
    factory, jax_stub, _, _ = _prepare_factory_with_stubs()

    gen = factory.create_data_generator(
        framework="jax",
        model="jax_model",
        dataset=("x", "y"),
        batch_size=8,
        device=None,
    )

    assert isinstance(gen, jax_stub)
    gen_stub = cast(_BaseStub, gen)
    assert gen_stub.model == "jax_model"
    assert gen_stub.dataset == ("x", "y")
    assert gen_stub.batch_size == 8
    assert gen_stub.device is None
