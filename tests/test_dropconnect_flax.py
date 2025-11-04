from __future__ import annotations

from flax import linen as nn
from jax import Array, random
import jax.numpy as jnp

from probly.transformation.dropconnect.common import dropconnect


class Tiny(nn.Module):
    """Minimal example model for DropConnect testing."""

    @nn.compact
    def __call__(self, x: Array, *, train: bool = True) -> Array:  # noqa: ARG002
        x = nn.Dense(4)(x)
        return nn.Dense(2)(x)

    @staticmethod
    def test_train_eval_diff() -> None:
        """Run a small test showing DropConnect behavior differences."""
        m = Tiny()
        md = dropconnect(m, p=0.5)

        x = jnp.ones((3, 5))
        key = random.PRNGKey(0)
        params = md.init(key, x, train=True)
        y1 = md.apply(params, x, train=True, rngs={"dropconnect": random.PRNGKey(1)})
        y2 = md.apply(params, x, train=True, rngs={"dropconnect": random.PRNGKey(2)})
        ye = md.apply(params, x, train=False)

        assert y1.shape == ye.shape == (3, 2)
        assert not jnp.allclose(y1, y2)
        assert not jnp.allclose(y1, ye)
