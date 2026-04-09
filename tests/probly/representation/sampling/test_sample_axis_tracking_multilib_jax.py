"""JAX-specific tests for planned multi-library axis tracking behavior."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
import jax
from jax import numpy as jnp

from probly.representation.sample.axis_tracking import track_axis


class TestArrayIndexSemantics:
    def test_jax_array_index_supported_in_numpy_jax_mode(self) -> None:
        idx = (0, slice(None), jnp.array([0, 2]))

        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0

    def test_jax_0d_integer_index_is_treated_like_python_int(self) -> None:
        idx = (jnp.array(0), slice(None), jnp.array([0, 2]))

        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0


class TestEllipsisSemantics:
    def test_ellipsis_with_nd_boolean_array_index(self) -> None:
        idx = (jnp.array([[True, False, True], [False, True, False]]), Ellipsis)

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=False) == 0
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 1

    def test_ellipsis_with_leading_newaxis_and_trailing_1d_integer_array_index(self) -> None:
        idx = (None, Ellipsis, jnp.array([0, 2]))

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 2
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 3

    def test_trailing_scalar_bool_with_ellipsis_keeps_last_axis(self) -> None:
        idx = (Ellipsis, jnp.array(True))

        assert track_axis(idx, special_axis=0, ndim=4, torch_indexing=False) == 0
        assert track_axis(idx, special_axis=1, ndim=4, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=2, ndim=4, torch_indexing=False) == 2
        assert track_axis(idx, special_axis=3, ndim=4, torch_indexing=False) == 3

    def test_empty_ellipsis_separates_advanced_indices(self) -> None:
        idx = (slice(None), jnp.array([0, 1]), Ellipsis, jnp.array([0, 1]))

        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0


class TestDeferredAdvancedIndexing:
    @pytest.mark.skip(reason="Mixed-rank advanced indexing behavior is intentionally deferred.")
    def test_mixed_rank_advanced_indexing_tracks_indexed_axis_position(self) -> None:
        idx = (jnp.array([[0, 1], [2, 0]]), slice(None), jnp.array([0, 2]), slice(None))

        assert track_axis(idx, special_axis=2, ndim=4, torch_indexing=False) == 1


class TestJittedIndexSemantics:
    def test_jitted_integer_array_index_is_supported(self) -> None:
        @jax.jit
        def _tracked_axis(idx: jax.Array) -> int:
            # Ensure this index type is accepted by jax itself.
            _ = jnp.zeros((2, 3, 4))[(0, slice(None), idx)]
            return track_axis((0, slice(None), idx), special_axis=1, ndim=3, torch_indexing=False)

        assert int(_tracked_axis(jnp.array([0, 2], dtype=jnp.int32))) == 1

    def test_jitted_0d_integer_index_is_supported(self) -> None:
        @jax.jit
        def _tracked_axis(idx: jax.Array) -> int:
            # Ensure this index type is accepted by jax itself.
            _ = jnp.zeros((2, 3, 4))[(idx, slice(None), jnp.array([0, 2], dtype=jnp.int32))]
            return track_axis((idx, slice(None), jnp.array([0, 2], dtype=jnp.int32)), special_axis=1, ndim=3)

        assert int(_tracked_axis(jnp.array(0, dtype=jnp.int32))) == 1
