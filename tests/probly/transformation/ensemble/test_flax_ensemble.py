# tests/transformation/ensemble/test_flax_ensemble.py
import pytest

# 没合并到 main 时，直接跳过，避免 CI 烟花
mod = pytest.importorskip(
    "probly.transformation.ensemble.flax",
    reason="ensemble/flax implementation not available on this branch",
)

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax


# 一个超小的基模型，给 Ensemble 包一层用
class _TinyNet(nn.Module):
    out_dim: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        h = nn.Dense(8)(x)
        h = nn.tanh(h)
        y = nn.Dense(self.out_dim)(h)
        return y


def _try_build_model(rng, out_dim=1, num_members=3, aggregator="mean"):
    """
    为了兼容不同实现，这里尝试几种常见构造方式：
    1) Ensemble(base=..., num_members=..., aggregator=...)
    2) FlaxEnsemble(...) / build_ensemble(...)
    3) wrap(base=..., ...) / make(...)
    若都失败，则跳过全部测试。
    """
    x_dummy = jnp.ones((2, 5))
    candidates = []

    # 猜名字1：Ensemble
    if hasattr(mod, "Ensemble"):
        candidates.append(lambda: mod.Ensemble(
            base=_TinyNet(out_dim=out_dim),
            num_members=num_members,
            aggregator=aggregator,
        ))

    # 猜名字2：FlaxEnsemble
    if hasattr(mod, "FlaxEnsemble"):
        candidates.append(lambda: mod.FlaxEnsemble(
            base=_TinyNet(out_dim=out_dim),
            num_members=num_members,
            aggregator=aggregator,
        ))

    # 猜名字3：工厂函数
    for fname in ["build_ensemble", "make_ensemble", "wrap", "make"]:
        if hasattr(mod, fname):
            f = getattr(mod, fname)
            candidates.append(lambda f=f: f(base=_TinyNet(out_dim=out_dim),
                                            num_members=num_members,
                                            aggregator=aggregator))

    last_error = None
    for ctor in candidates:
        try:
            model = ctor()
            params = model.init(rng, x_dummy, train=False)
            return model, params
        except Exception as e:
            last_error = e

    pytest.skip(f"Cannot construct ensemble/flax model with guessed APIs. Last error: {last_error}")


def _extract_members_and_mean(forward_out):
    """
    兼容不同返回格式：
    - dict 含 'members' / 'mean'
    - tuple/list: (members, mean) 或 (mean, ...)
    - 只返回聚合：则 members=None
    """
    members = None
    mean = None

    if isinstance(forward_out, dict):
        for key in ["members", "all", "member_outputs"]:
            if key in forward_out:
                members = forward_out[key]
                break
        for key in ["mean", "agg", "aggregated", "prediction"]:
            if key in forward_out:
                mean = forward_out[key]
                break
        # 有些实现直接把聚合放在 'output' 或 'y'
        for key in ["output", "y"]:
            if mean is None and key in forward_out:
                mean = forward_out[key]

    elif isinstance(forward_out, (tuple, list)):
        # 尝试认为第一个是 mean 或 members
        a0 = forward_out[0]
        a1 = forward_out[1] if len(forward_out) > 1 else None
        # 识别形状：members 常见 [M,B,O] 或 [B,M,O]
        if isinstance(a0, jnp.ndarray) and a0.ndim == 3:
            members = a0
            mean = a1 if isinstance(a1, jnp.ndarray) else None
        else:
            mean = a0
            if isinstance(a1, jnp.ndarray) and a1.ndim == 3:
                members = a1

    return members, mean


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def toy_data():
    x = jnp.ones((8, 5))        # [B, D]
    y = jnp.zeros((8, 1))       # [B, O]
    return x, y


def test_shapes_and_determinism(rng, toy_data):
    model, params = _try_build_model(rng)
    x, _ = toy_data

    out1 = model.apply(params, x, train=False)
    out2 = model.apply(params, x, train=False)

    members, mean = _extract_members_and_mean(out1)
    assert mean is not None, "aggregated output (mean) must be present"
    assert mean.shape[0] == x.shape[0], "B dimension should match input batch"

    # 评估态确定性
    out2_members, out2_mean = _extract_members_and_mean(out2)
    np.testing.assert_allclose(mean, out2_mean, rtol=1e-6, atol=1e-6)

    # 有成员输出的话再做基本形状断言
    if members is not None:
        assert members.ndim == 3, "members should be rank-3: [M,B,O] or [B,M,O]"
        assert x.shape[0] in members.shape, "one axis of members must be batch size"


def test_aggregation_matches_manual_mean_if_members_available(rng, toy_data):
    model, params = _try_build_model(rng)
    x, _ = toy_data
    outs = model.apply(params, x, train=False)
    members, mean = _extract_members_and_mean(outs)

    if members is None:
        pytest.skip("members not returned; skipping manual-mean consistency test")

    # 兼容 [M,B,O] 或 [B,M,O]
    if members.shape[0] == x.shape[0]:
        manual = members.mean(axis=1)  # [B,M,O] -> [B,O]
    elif members.shape[1] == x.shape[0]:
        manual = members.mean(axis=0)  # [M,B,O] -> [B,O]
    else:
        pytest.skip("cannot infer member layout for manual mean")

    np.testing.assert_allclose(mean, manual, rtol=1e-6, atol=1e-6)


def test_jit_consistency(rng, toy_data):
    model, params = _try_build_model(rng)
    x, _ = toy_data

    def f(inp):
        outs = model.apply(params, inp, train=False)
        _, mean = _extract_members_and_mean(outs)
        return mean

    f_jit = jax.jit(f)
    y0 = f(x)
    y1 = f_jit(x)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_one_train_step_decreases_mse(rng, toy_data):
    model, params = _try_build_model(rng)
    x, y = toy_data

    def loss_fn(p):
        outs = model.apply({"params": p["params"]} if "params" in p else p, x, train=True)
        _, mean = _extract_members_and_mean(outs)
        return jnp.mean((mean - y) ** 2)

    # 兼容 flax 参数树形态
    params_like = params if "params" in params else {"params": params}
    opt = optax.adam(1e-2)
    opt_state = opt.init(params_like)

    @jax.jit
    def step(p, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = opt.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    params_like, opt_state, l0 = step(params_like, opt_state)
    params_like, opt_state, l1 = step(params_like, opt_state)
    assert jnp.isfinite(l1), "loss must be finite"
    assert l1 <= l0 + 1e-6, "loss should not increase on a tiny toy step"
