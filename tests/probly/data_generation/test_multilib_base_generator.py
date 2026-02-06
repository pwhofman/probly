from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from probly.data_generation.base_generator import BaseDataGenerator


# A minimal concrete subclass relying on BaseDataGenerator's defaults
class DummyDataGenerator(BaseDataGenerator):
    def generate(self) -> dict[str, Any]:
        return {"sample_0": [0.1, 0.9], "sample_1": [0.3, 0.7]}


def test_base_generator_init_and_get_info() -> None:
    gen = DummyDataGenerator(model="fake", dataset=[], batch_size=16, device="cpu")

    assert gen.batch_size == 16
    assert gen.device == "cpu"
    assert gen.get_info() == {"batch_size": 16, "device": "cpu"}


def test_abstract_methods_enforced() -> None:
    class Incomplete(BaseDataGenerator):
        pass

    with pytest.raises(TypeError):
        Incomplete(model=None, dataset=[])


def test_generate_save_load_roundtrip(tmp_path: Path) -> None:
    gen = DummyDataGenerator(
        model=None,
        dataset=[],
        batch_size=8,
        device="test",
    )

    # Generate some fake data
    results = gen.generate()

    # Save to temp path
    save_path = tmp_path / "test.json"
    gen.save(str(save_path))

    # Load back
    loaded = gen.load(str(save_path))
    assert loaded == results


def test_save_writes_results_only(tmp_path: Path) -> None:
    gen = DummyDataGenerator(model="test_model", dataset=["a", "b"], batch_size=32, device=None)

    save_path = tmp_path / "results_only.json"
    gen.save(str(save_path))

    with save_path.open() as f:
        data = json.load(f)

    # BaseDataGenerator.save writes only the results dict
    assert isinstance(data, dict)
    assert data == gen.generate()
