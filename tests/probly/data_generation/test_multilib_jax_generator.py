import os
import numpy as np
import pytest


def _has_jax():
    """Used to check whether JAX is installed in the runtime environment; if not, skip the entire test suite to avoid CI failures."""
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        return True
    except Exception:
        return False


# -----------------------------
# [Environment robustness]: if JAX is not available, "skip" instead of "fail"
# Purpose:
# - Allow CI / collaborator environments (without JAX installed) to run the full test suite
# - Avoid ImportError causing the entire test suite to fail
# -----------------------------
pytestmark = pytest.mark.skipif(not _has_jax(), reason="jax is not installed")


# -----------------------------
# [Functional correctness / main path] Round-trip save–load cycle test
# Purpose:
# 1) Verify that the round-trip from JAX array dict -> save (.npz) -> load -> JAX array dict is reversible
# 2) Verify key invariants: key set, shape, dtype, and values remain identical (baseline for scientific reproducibility)
# 3) Also cover: automatic .npz suffix appending, automatic directory creation, and device_put to a specified device
# -----------------------------
def test_jax_generator_save_load_roundtrip(tmp_path):
    import jax
    import jax.numpy as jnp

    # According to your project structure: from <pkg>.jax_generator import JAXGenerator
    # Here we use relative import: assuming the test and generator are in the same package
    from .jax_generator import JAXGenerator

    gen = JAXGenerator()

    tensor_dict = {
        "a": jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        "b": jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
    }

    # Intentionally omit the suffix to test that _ensure_suffix automatically appends .npz
    save_path = tmp_path / "dist"
    gen.save_distributions(
        tensor_dict=tensor_dict,
        save_path=str(save_path),
        create_dir=True,   # Coverage: directory should be created automatically if it does not exist
        verbose=False,
    )

    expected_file = str(save_path) + ".npz"
    assert os.path.exists(expected_file)

    # Coverage: explicit device parameter (even if only CPU is available, device_put should still work)
    loaded = gen.load_distributions(load_path=expected_file, device="cpu:0", verbose=False)

    # Key set consistency
    assert set(loaded.keys()) == set(tensor_dict.keys())

    # Shape / dtype / value consistency
    for k in tensor_dict.keys():
        x = np.asarray(jax.device_get(tensor_dict[k]))
        y = np.asarray(jax.device_get(loaded[k]))
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert np.array_equal(x, y)


# -----------------------------
# [Interface contract / input validation] keys must be str
# Purpose:
# - np.savez_compressed requires **kwargs, therefore keys must be valid keyword / field names (str)
# - Fail early: provide a clear error at the generator level instead of a hard-to-trace numpy error
# -----------------------------
def test_jax_generator_rejects_non_mapping_keys(tmp_path):
    import jax.numpy as jnp
    from .jax_generator import JAXGenerator

    gen = JAXGenerator()

    bad = {123: jnp.array([1, 2, 3])}  # key is not str
    with pytest.raises(TypeError):
        gen.save_distributions(bad, str(tmp_path / "x"), verbose=False)


# -----------------------------
# [Backend isolation / type safety] values must be JAX arrays
# Purpose:
# - Make the semantics of JAXGenerator explicit: only jax.Array is supported (avoid silent bugs from mixing numpy/torch)
# - Ensure device_get / device_put behavior is well-defined and avoid cross-framework data contamination
# -----------------------------
def test_jax_generator_rejects_non_jax_values(tmp_path):
    from .jax_generator import JAXGenerator

    gen = JAXGenerator()

    bad = {"a": np.array([1, 2, 3])}  # value is not a JAX array (it is numpy)
    with pytest.raises(TypeError):
        gen.save_distributions(bad, str(tmp_path / "x"), verbose=False)


# -----------------------------
# [Clear failure modes] loading a non-existent file must raise FileNotFoundError
# Purpose:
# - Common error scenarios: incorrect path, missing artifacts, lost mounted volumes
# - Fail early with a clear error to reduce debugging cost
# --------sssss---------------------
def test_jax_generator_load_missing_file(tmp_path):
    from .jax_generator import JAXGenerator

    gen = JAXGenerator()

    with pytest.raises(FileNotFoundError):
        gen.load_distributions(str(tmp_path / "not_exists.npz"), verbose=False)


# -----------------------------
# [Defensive input checks] load path must have .npz suffix
# Purpose:
# - Prevent users from mistakenly loading .pt/.npy/.ckpt/.tif etc. as distribution files
# - Fail early with a readable error instead of obscure numpy exceptions
# -----------------------------
def test_jax_generator_load_requires_npz_suffix(tmp_path):
    from .jax_generator import JAXGenerator

    gen = JAXGenerator()

    p = tmp_path / "bad.ext"
    p.write_text("not a npz")

    with pytest.raises(ValueError):
        gen.load_distributions(str(p), verbose=False)