# tests/probly/transformation/ensemble/test_flax_ensemble.py
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# 你们真实的 API：src/probly/transformation/ensemble/flax.py::generate_flax_ensemble
_mod = pytest.importorskip(
    "probly.transformation.ensemble.flax",
    reason="ensemble/flax 实现未在当前分支提供",
)
generate_flax_ensemble = _mod.generate_flax_ensemble

# 直接用仓库自带的小模型 fixture 当 base
from tests.probly.fixtures.flax_models import flax_model_small_2d_2d


# 统一前向：NNX 直接 model(x)；只有 linen 才会有 .apply
def _fwd(model, x):
    # 1) nnx：Linear/Sequential 都是这样用
    try:
        return model(x)
    except Exception:
        pass
    # 2) linen：才存在 .apply
    if hasattr(model, "apply"):
        try:
            return model.apply(x)
        except Exception:
            pass
    raise AssertionError("前向调用方式无法匹配（既不是 nnx 也不是 linen）")


def _to_array(out):
    """
    把各种返回格式捏成 ndarray：
    - ndarray: 直接返回
    - dict: 依次取 'mean'/'output'/'y'/'agg'/'aggregated'，取不到就找第一个 ndarray 值
    - tuple/list: 取第一个 ndarray
    """
    if isinstance(out, jnp.ndarray):
        return np.asarray(out)
    if isinstance(out, dict):
        for k in ("mean", "output", "y", "agg", "aggregated"):
            v = out.get(k, None)
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)
        for v in out.values():
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)
    raise AssertionError("无法从前向输出中提取 ndarray，用例需要数值输出进行比较")


@pytest.fixture
def xbatch() -> jnp.ndarray:
    # 你的小模型是 2->2 的线性堆栈。别喂 5 维进去。
    return jnp.ones((8, 2), dtype=jnp.float32)


def test_returns_sequence_and_types(flax_model_small_2d_2d, xbatch):
    """
    - 返回序列，长度等于 num_members
    - 每个成员类型与 base 一致
    - 成员不是同一个对象引用
    - 单成员前向形状与 base 一致
    """
    base = flax_model_small_2d_2d
    num = 3
    members = generate_flax_ensemble(base, num_members=num, reset_params=False)

    assert isinstance(members, (list, tuple)), "应返回一个成员序列"
    assert len(members) == num

    ids = set()
    y_base = _to_array(_fwd(base, xbatch))
    for i, m in enumerate(members):
        assert isinstance(m, type(base)), f"第 {i} 个成员的类型应与 base 相同"
        ids.add(id(m))
        y_i = _to_array(_fwd(m, xbatch))
        assert y_i.shape == y_base.shape, "成员输出形状应与 base 输出一致"

    assert len(ids) == num, "每个成员应是不同对象（不是同一引用）"


def test_reset_params_false_outputs_identical(flax_model_small_2d_2d, xbatch):
    """
    reset_params=False：拷贝参数不重置，同一输入应得到一致输出。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=False)

    outs = [_to_array(_fwd(m, xbatch)) for m in members]
    for i in range(1, len(outs)):
        np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)


def test_reset_params_true_outputs_different(flax_model_small_2d_2d, xbatch):
    """
    reset_params=True：每个成员用不同 PRNGKey 重新初始化，
    合理情况下至少存在一对成员输出不同。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=True)

    outs = [_to_array(_fwd(m, xbatch)) for m in members]
    diff_pairs = 0
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            if not np.allclose(outs[i], outs[j], rtol=1e-6, atol=1e-6):
                diff_pairs += 1
    assert diff_pairs >= 1, "reset_params=True 时，至少应有一对成员输出不相同"


def test_reset_params_true_is_deterministic_across_calls(flax_model_small_2d_2d, xbatch):
    """
    你们实现里 reset 用 PRNGKey(i)，因此两次生成应可复现（按索引逐一对应）。
    """
    base = flax_model_small_2d_2d
    num = 3
    ens1 = generate_flax_ensemble(base, num_members=num, reset_params=True)
    ens2 = generate_flax_ensemble(base, num_members=num, reset_params=True)

    for i in range(num):
        y1 = _to_array(_fwd(ens1[i], xbatch))
        y2 = _to_array(_fwd(ens2[i], xbatch))
        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_member_forward_jit_consistency(flax_model_small_2d_2d, xbatch):
    """
    把“成员前向”包一层 jit（闭包捕获模型，不把 Python 对象当参数传给 jitted 函数），
    确保 jit 前后结果一致。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=2, reset_params=False)
    m0 = members[0]

    def f(x):
        return _to_array(_fwd(m0, x))

    f_jit = jax.jit(f)
    y0 = f(xbatch)
    y1 = f_jit(xbatch)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)
