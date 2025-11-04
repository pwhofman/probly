from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# 1) 明确按你的实现导入（别再猜 API 了）
from probly.transformation.ensemble.flax import generate_flax_ensemble

# 用你们现成的小模型 fixture，当作 base
from tests.probly.fixtures.flax_models import flax_model_small_2d_2d


def _fwd(m, x, train=False):
    """
    统一的前向调用：兼容 nnx/linen 常见两套写法。
    """
    try:
        return m(x, train=train)
    except TypeError:
        # linen 风格
        try:
            return m.apply(x, train=train)
        except TypeError:
            return m.apply(x)  # 某些实现不吃 train kw


@pytest.fixture
def xbatch():
    # 你们的小模型基本吃 [B, D]；维度和 fixtures 对齐
    return jnp.ones((8, 5))


def test_returns_correct_length_and_types(flax_model_small_2d_2d, xbatch):
    """
    生成的 ensemble：
      - 长度等于 num_members
      - 成员类型与 base 一致
      - 成员对象不是同一个引用
      - 前向形状与单模型一致
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=3, reset_params=False)

    assert isinstance(members, (list, tuple)), "generate_flax_ensemble 应该返回一个序列"
    assert len(members) == 3

    # 类型一致且不是同一对象
    for i, m in enumerate(members):
        assert isinstance(m, type(base)), f"第 {i} 个成员类型应与 base 相同"
    assert len({id(m) for m in members}) == len(members), "成员对象不应是同一引用"

    # 形状 sanity：与 base 的输出形状一致
    y_base = _fwd(base, xbatch, train=False)
    for i, m in enumerate(members):
        y_i = _fwd(m, xbatch, train=False)
        assert y_i.shape == y_base.shape, f"第 {i} 个成员的输出形状应与 base 一致"


def test_reset_params_false_members_outputs_are_identical(flax_model_small_2d_2d, xbatch):
    """
    reset_params=False：成员是“拷贝”而非重新初始化，同一输入输出应一致。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=3, reset_params=False)

    outs = [np.asarray(_fwd(m, xbatch, train=False)) for m in members]
    for i in range(1, len(outs)):
        np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)


def test_reset_params_true_members_outputs_are_different(flax_model_small_2d_2d, xbatch):
    """
    reset_params=True：每个成员参数用不同 PRNGKey 重新初始化，同一输入输出应有差异。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=3, reset_params=True)

    outs = [np.asarray(_fwd(m, xbatch, train=False)) for m in members]
    # 只要存在任意一对不相等即可（避免过度苛刻）
    different_pairs = 0
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            if not np.allclose(outs[i], outs[j], rtol=1e-6, atol=1e-6):
                different_pairs += 1
    assert different_pairs >= 1, "reset_params=True 时，不同成员的输出应当存在差异"


def test_reset_params_true_is_deterministic_across_calls(flax_model_small_2d_2d, xbatch):
    """
    在你们实现里，reset 后的种子按 PRNGKey(i) 固定。
    因此同样的 base、相同 num_members、相同 reset=True 的两次生成应可复现（成员位次一一对应）。
    """
    base = flax_model_small_2d_2d

    ens1 = generate_flax_ensemble(base, num_members=3, reset_params=True)
    ens2 = generate_flax_ensemble(base, num_members=3, reset_params=True)

    outs1 = [np.asarray(_fwd(m, xbatch, train=False)) for m in ens1]
    outs2 = [np.asarray(_fwd(m, xbatch, train=False)) for m in ens2]

    assert len(outs1) == len(outs2) == 3
    for i in range(3):
        np.testing.assert_allclose(outs1[i], outs2[i], rtol=1e-6, atol=1e-6)


def test_member_forward_under_jit_consistency(flax_model_small_2d_2d, xbatch):
    """
    把“单个成员的前向”包一层 jit，确保 jit 前后结果一致。
    """
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=2, reset_params=False)

    def f(m, x):
        return _fwd(m, x, train=False)

    f_jit = jax.jit(f)

    y0 = f(members[0], xbatch)
    y1 = f_jit(members[0], xbatch)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)

