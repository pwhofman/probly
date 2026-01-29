from __future__ import annotations

import importlib.util as importlib_util
from pathlib import Path

import pytest


def _has_jax():
    """Return True if JAX is available; otherwise False."""
    return (importlib_util.find_spec("jax") is not None) and (importlib_util.find_spec("jax.numpy") is not None)


# If JAX is not available, skip this modules tests
pytestmark = pytest.mark.skipif(not _has_jax(), reason="jax is not installed")


def _get_jax_data_generator_or_skip(required_methods: tuple[str, ...] = ()):
    """Import JAXDataGenerator and skip the test if methods are missing.

    Some environments provide a stub without full API. This helper guards each
    test individually using hasattr checks and calls pytest.skip when needed.
    """
    try:
        from probly.data_generation.jax_generator import JAXDataGenerator  # noqa: PLC0415
    except (ModuleNotFoundError, ImportError):
        pytest.skip("JAXDataGenerator import failed")
    for m in required_methods:
        if not hasattr(JAXDataGenerator, m):
            pytest.skip(f"JAXDataGenerator missing required method: {m}")
    return JAXDataGenerator


def _make_dummy_model():
    import jax.numpy as jnp  # noqa: PLC0415

    def model(x: jnp.ndarray) -> jnp.ndarray:
        # Simple linear logits: two classes, logits favor class 1 when sum(x) > 0
        s = jnp.sum(x, axis=1, keepdims=True)
        return jnp.concatenate([-s, s], axis=1)

    return model


def test_jax_generator_generate_and_metrics():
    import jax.numpy as jnp  # noqa: PLC0415

    jax_data_generator_cls = _get_jax_data_generator_or_skip(("generate",))

    x = jnp.array([[1.0, 0.0], [-1.0, 0.0], [2.0, -1.0], [0.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
    gen = jax_data_generator_cls(model=_make_dummy_model(), dataset=(x, y), batch_size=2)

    results = gen.generate()

    # Basic structure checks
    assert "info" in results
    assert "metrics" in results
    assert "class_distribution" in results
    assert "confidence" in results
    assert results["info"]["framework"] == "jax"
    assert results["info"]["dataset_size"] == 4
    assert results["info"]["batch_size"] == 2

    # Accuracy should be in [0, 1]
    acc = results["metrics"]["accuracy"]
    assert isinstance(acc, float)
    assert acc >= 0.0
    assert acc <= 1.0


def test_jax_generator_save_load_roundtrip(tmp_path):
    import jax.numpy as jnp  # noqa: PLC0415

    jax_data_generator_cls = _get_jax_data_generator_or_skip(("generate", "save", "load"))

    x = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([1, 0], dtype=jnp.int32)
    gen = jax_data_generator_cls(model=_make_dummy_model(), dataset=(x, y))

    saved = gen.generate()
    save_path = tmp_path / "results.json"
    gen.save(str(save_path))
    assert Path(save_path).exists()

    loaded = gen.load(str(save_path))

    for key in ("info", "metrics", "class_distribution", "confidence"):
        assert key in loaded
        assert key in saved


def test_jax_generator_load_missing_file(tmp_path):
    jax_data_generator_cls = _get_jax_data_generator_or_skip(("load",))

    gen = jax_data_generator_cls(model=_make_dummy_model(), dataset=([], []))

    # The current implementation swallows errors and returns {} on failure
    missing = tmp_path / "not_exists.json"
    out = gen.load(str(missing))
    assert out == {}


def test_jax_generator_save_without_generate_is_noop(tmp_path):
    jax_data_generator_cls = _get_jax_data_generator_or_skip(("save",))

    gen = jax_data_generator_cls(model=_make_dummy_model(), dataset=([], []))
    # Make sure the result is noop, doesnt produce artifacts (as per request)
    path = tmp_path / "noop.json"
    gen.save(str(path))
    assert not path.exists()
