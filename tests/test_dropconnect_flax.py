import jax
import jax.numpy as jnp
from flax import linen as nn
from probly.transformation.dropconnect.common import dropconnect

class Tiny(nn.Module):
    @nn.compact
    def __call__(self, x, *, train: bool = True):
        x = nn.Dense(4)(x)
        return nn.Dense(2)(x)
    
    def test_train_eval_diff():
        m = Tiny()
        md = dropconnect(m, p=0.5)
        x = jnp.ones((3, 5))
        key = jax.random.PRNGKey(0)
        vars = md.init(key, x, train=True)
        y1 = md.apply(vars, x, train=True, rngs={"dropconnect": jax.random.PRNGKey(1)})
        y2 = md.apply(vars, x, train=True, rngs={"dropconnect": jax.random.PRNGKey(2)})
        ye = md.apply(vars, x, train=False)
        assert y1.shape == ye.shape == (3, 2)
        assert not jnp.allclose(y1, y2)
        assert not jnp.allclose(y1, ye)

    
    assert 1 == 1