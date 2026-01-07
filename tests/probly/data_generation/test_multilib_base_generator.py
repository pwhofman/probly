from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from probly.data_generation.base_generator import BaseDataGenerator


class SavePayload(TypedDict):
    results: dict[str, Any]
    info: dict[str, Any]


# A minimal concrete subclass for testing the base class
class DummyDataGenerator(BaseDataGenerator):
    def generate(self) -> dict[str, Any]:
        return {"sample_0": [0.1, 0.9], "sample_1": [0.3, 0.7]}

    def save(self, path: str) -> None:
        data: SavePayload = {
            "results": self.generate(),
            "info": self.get_info(),
        }
        with Path(path).open("w") as f:
            json.dump(data, f)

    def load(self, path: str) -> dict[str, Any]:
        with Path(path).open() as f:
            data = cast(SavePayload, json.load(f))
        return data["results"]


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


def test_save_includes_base_info(tmp_path: Path) -> None:
    gen = DummyDataGenerator(model="test_model", dataset=["a", "b"], batch_size=32, device=None)

    save_path = tmp_path / "with_info.json"
    gen.save(str(save_path))

    with save_path.open() as f:
        data = json.load(f)

    assert data["info"] == gen.get_info()
    assert data["info"]["batch_size"] == 32
    assert data["info"]["device"] is None
