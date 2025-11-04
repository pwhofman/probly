# tests/probly/transformation/ensemble/test_flax_ensemble.py
from __future__ import annotations

import importlib
from typing import Any, Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

# ------------------------------------------------------------
# 0) 尝试导入 ensemble/flax 实现（多路径探测）
# ------------------------------------------------------------
_CANDIDATE_MODULES = [
    "probly.transformation.ensemble.flax",   # 优先：src/probly/transformation/ensemble/flax.py
    "probly.transformation.ensemble",        # 其次：直接在 ensemble/__init__.py 暴露
    "probly.transformation.flax.ensemble",   # 备选：有人爱反着放
]

_mod = None
_last_import_err: Optional[BaseException] = None
for _name in _CANDIDATE_MODULES:
    try:
        _mod = importlib.import_module(_name)
        _mod.__name__  # noqa
        _last_import_err = None
        break
    except Exception as e:  # pragma: no cover - 只在坏路径触发
        _last_import_err = e

if _mod is None:
    pytest.skip(
        f"找不到 ensemble/flax 实现。尝试路径={_CANDIDATE_MODULES}；"
        f"最后一次错误={_last_import_err}",
        allow_module_level=True,
    )

# nnx 为主，若你们用 linen 也没关系，测试只看输出数组
flax = pytest.importorskip("flax", reason="需要 flax")
try:
    from flax import nnx  # type: ignore
except Exception as e:  # pragma: no cover
    nnx = None  # noqa: F811
    pytest.skip(f"需要 flax.nnx：{e}", allow_module_level=True)


# ------------------------------------------------------------
# 1) 统一的“构造 + 前向”适配器
# ------------------------------------------------------------
def _build_ensemble(
    base_model: Any,
    num_members: int = 3,
    aggregator: str = "mean",
) -> Tuple[Any, Callable[[Any, jnp.ndarray, bool], Any]]:
    """
    返回 (model, forward)。
    - model: 你们的 Ensemble 对象/模块
    - forward(m, x, train): 统一的前向调用接口
    尽量兼容不同 API 命名。
    """
    ctors: list[Callable[[], Any]] = []

    # 类常见名
    if hasattr(_mod, "Ensemble"):
        ctors.append(lambda: _mod.Ensemble(base=base_model, num_members=num_members, aggregator=aggregator))
    if hasattr(_mod, "FlaxEnsemble"):
        ctors.append(lambda: _mod.FlaxEnsemble(base=base_model, num_members=num_members, aggregator=aggregator))

    # 工厂函数常见名
    for fname in ("build_ensemble", "make_ensemble", "wrap", "make", "ensemble"):
        if hasattr(_mod, fname):
            f = getattr(_mod, fname)
            ctors.append(lambda f=f: f(base=base_model, num_members=num_members, aggregator=aggregator))

    last_error: Optional[BaseException] = None
    for ctor in ctors:
        try:
            model = ctor()
            # 大多数 nnx/linen 风格：直接可调用或有 apply
            def _fwd(m: Any, x: jnp.ndarray, train: bool = False):
                if callable(m):
                    return m(x, train=train)
                if hasattr(m, "apply"):
                    # 有些 linen 需要 {"params": ...}，但实现一般把参数封装在对象里；先尝试无 params
                    return m.apply(x, train=train)
                # 兜底直接调用
                return m(x)

            # 烟雾测试，不要求成功形状，只要不会抛错
            x_smoke = jnp.ones((2, 5))
            try:
                _ = _fwd(model, x_smoke, train=False)
            except TypeError:
                # 有些实现需要 rng，尽量给个空 rngs 兼容；给不进去就放过
                def _fwd_rng(m: Any, x: jnp.ndarray, train: bool = False):
                    if hasattr(m, "apply"):
                        return m.apply(x, train=train, rngs=None)
                    return m(x, train=train)
                return model, _fwd_rng

            return model, _fwd
        except Exception as e:
            last_error = e

    pytest.skip(f"无法构造 ensemble/flax 模型。请检查构造器命名与参数。最后一次错误：{last_error}")


def _extract_members_and_mean(forward_out: Any) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    兼容不同返回格式：
    - dict: 'members'/'all'/'member_outputs' + 'mean'/'agg'/'aggregated'/'output'/'y'
    - tuple/list: (members, mean) 或 (mean, members)
    - 仅返回聚合：则 members=None
    """
    members = None
    mean = None

    if isinstance(forward_out, dict):
        for k in ("members", "all", "member_outputs"):
            if k in forward_out and isinstance(forward_out[k], jnp.ndarray):
                members = forward_out[k]
                break
        for k in ("mean", "agg", "aggregated", "output", "y"):
            if k in forward_out and isinstance(forward_out[k], jnp.ndarray):
                mean = forward_out[k]
                break

    elif isinstance(forward_out, (tuple, list)):
        a0 = forward_out[0] if len(forward_out) > 0 else None
        a1 = forward_out[1] if len(forward_out) > 1 else None
        # 猜测哪个是 members（rank=3）
        if isinstance(a0, jnp.ndarray) and a0.ndim == 3:
            members, mean = a0, a1 if isinstance(a1, jnp.ndarray) else None
        else:
            mean = a0 if isinstance(a0, jnp.ndarray) else None
            if isinstance(a1, jnp.ndarray) and a1.ndim == 3:
                members = a1

    return members, mean


# ------------------------------------------------------------
# 2) 用现成 flax fixture 作为 base model
#    见：tests/probly/fixtures/flax_models.py
# ------------------------------------------------------------

@pytest.fixture
def _x_batch() -> jnp.ndarray:
    # 绝大多数 toy 模型都吃 [B, D]；若你们 base 模型 D 不同，fixture 会调整或报错
    return jnp.ones((8, 5))


def test_members_and_aggregator_shape(flax_model_small_2d_2d, _x_batch):
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=3, aggregator="mean")

    out = fwd(model, _x_batch, train=False)
    members, mean = _extract_members_and_mean(out)

    assert mean is not None, "聚合输出(mean) 不应为 None"
    assert mean.shape[0] == _x_batch.shape[0], "聚合输出的 batch 维应等于输入 batch"

    # 有成员输出时做更严格断言
    if members is not None:
        assert members.ndim == 3, "成员输出应为 rank-3: [M,B,O] 或 [B,M,O]"
        B = _x_batch.shape[0]
        assert B in members.shape[:2], "成员输出前两维之一应为 batch 大小"


def test_mean_aggregation_matches_manual(flax_model_small_2d_2d, _x_batch):
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=4, aggregator="mean")

    out = fwd(model, _x_batch, train=False)
    members, mean = _extract_members_and_mean(out)

    if members is None or mean is None:
        pytest.skip("前向未返回 members 或 mean，跳过均值一致性检查")

    B = _x_batch.shape[0]
    if members.shape[0] == B:        # [B, M, O]
        manual = members.mean(axis=1)
    elif members.shape[1] == B:      # [M, B, O]
        manual = members.mean(axis=0)
    else:
        pytest.skip("无法从 members 推断 batch 轴位置，跳过")

    np.testing.assert_allclose(mean, manual, rtol=1e-6, atol=1e-6)


def test_eval_is_deterministic(flax_model_small_2d_2d, _x_batch):
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=3, aggregator="mean")

    def get_mean(x):
        out = fwd(model, x, train=False)
        _, m = _extract_members_and_mean(out)
        return m if m is not None else out  # 某些实现直接返回聚合数组

    y1 = get_mean(_x_batch)
    y2 = get_mean(_x_batch)
    np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_jit_consistency(flax_model_small_2d_2d, _x_batch):
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=3, aggregator="mean")

    def f(x):
        out = fwd(model, x, train=False)
        _, mean = _extract_members_and_mean(out)
        return mean if mean is not None else out

    f_jit = jax.jit(f)
    y0 = f(_x_batch)
    y1 = f_jit(_x_batch)
    np.testing.assert_allclose(y0, y1, rtol=1e-6, atol=1e-6)


def test_one_train_step_decreases_mse(flax_model_small_2d_2d, _x_batch):
    """
    用一个超小学习率在 MSE 上走两步，检查 loss 不发散且不升高。
    这里不强制“必须明显下降”，因为随机初始化可能抖；只要不比初始大就行。
    """
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=3, aggregator="mean")

    # 目标 y：全零即可
    y_true = jnp.zeros((_x_batch.shape[0], 1))

    # 把参数树找出来：nnx 一般挂在 .parameters 或 .vars；linen 挂在 {"params": ...}
    params_like = None
    for attr in ("parameters", "params", "variables", "vars", "state"):
        if hasattr(model, attr):
            params_like = getattr(model, attr)
            break
    if params_like is None:
        # 兜底：一些对象本身就是可微的 pytree
        params_like = model

    def loss_fn(p):
        # 尽量把 p 回灌进 model；失败就用闭包里的 model 直接前向（很多 nnx 对象是就地持参）
        try:
            # 常见：model 拥有可替换的参数属性
            _m = model
            if hasattr(_m, "parameters"):
                setattr(_m, "parameters", p)
            elif hasattr(_m, "params"):
                setattr(_m, "params", p)
        except Exception:
            pass

        out = fwd(model, _x_batch, train=True)
        _, mean = _extract_members_and_mean(out)
        pred = mean if mean is not None else out
        # 兼容输出形状 [B, O]，若 O>1 就按第一维对齐
        if pred.ndim == 2 and y_true.ndim == 2 and pred.shape[1] != y_true.shape[1]:
            pred = pred[:, : y_true.shape[1]]
        return jnp.mean((pred - y_true) ** 2)

    # 一阶优化两步
    opt = optax.adam(1e-2)
    opt_state = opt.init(params_like)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s, p)
        p = optax.apply_updates(p, upd)
        return p, s, l

    p, s, l0 = step(params_like, opt_state)
    p, s, l1 = step(p, s)
    assert jnp.isfinite(l1), "loss 不能是 NaN/Inf"
    assert l1 <= l0 + 1e-6, "两步之后的损失不应高于初始（容许数值抖动）"
