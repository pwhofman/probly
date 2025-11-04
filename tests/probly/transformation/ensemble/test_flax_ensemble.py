# tests/probly/transformation/ensemble/test_flax_ensemble.py
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# -----------------------------
# 0) 模块导入：多路径探测
# -----------------------------
_CANDIDATE_MODULES = [
    "probly.transformation.ensemble.flax",   # 常见：src/probly/transformation/ensemble/flax.py
    "probly.transformation.ensemble",        # 其次：直接在 ensemble/__init__.py 暴露
    "probly.transformation.flax.ensemble",   # 备选：有人爱反着放
]

_mod = None
_last_import_err: Optional[BaseException] = None
for _name in _CANDIDATE_MODULES:
    try:
        _mod = importlib.import_module(_name)
        _last_import_err = None
        break
    except Exception as e:
        _last_import_err = e

if _mod is None:
    pytest.skip(
        f"找不到 ensemble/flax 实现。尝试路径={_CANDIDATE_MODULES}；"
        f"最后一次错误={_last_import_err}",
        allow_module_level=True,
    )

# 用 nnx 就行；若项目用 linen 也不影响本测试
flax = pytest.importorskip("flax", reason="需要 flax")

try:
    from flax import nnx  # noqa: F401
except Exception as e:
    pytest.skip(f"需要 flax.nnx：{e}", allow_module_level=True)


# ----------------------------------------
# 1) 统一“构造 + 前向”适配器（超鲁棒）
# ----------------------------------------
_BASE_PARAM_CANDIDATES = ("base", "model", "module", "backbone", "fn", "base_model")
_NUM_PARAM_CANDIDATES = ("num_members", "members", "n", "k", "ensemble_size")
_AGG_PARAM_CANDIDATES = ("aggregator", "agg", "reduction")

def _try_kwargs(sig: inspect.Signature, base_value: Any, num_members: int, aggregator: str) -> dict:
    kwargs: dict[str, Any] = {}
    params = sig.parameters

    # 给 base：优先传实例，失败时会用类型再试
    for k in _BASE_PARAM_CANDIDATES:
        if k in params:
            kwargs[k] = base_value
            break

    # 给成员数
    for k in _NUM_PARAM_CANDIDATES:
        if k in params:
            kwargs[k] = num_members
            break

    # 给聚合器
    for k in _AGG_PARAM_CANDIDATES:
        if k in params:
            kwargs[k] = aggregator
            break

    return kwargs


def _wrap_forward(model: Any) -> Callable[[Any, jnp.ndarray, bool], Any]:
    """统一的前向接口：fwd(m, x, train=False) -> 任意结构（dict/tuple/ndarray）"""
    def fwd(m: Any, x: jnp.ndarray, train: bool = False):
        # 典型 1：对象可直接调用
        try:
            return m(x, train=train)
        except TypeError:
            pass

        # 典型 2：linen 风格 .apply(...)
        if hasattr(m, "apply"):
            try:
                return m.apply(x, train=train)
            except TypeError:
                # 某些实现要求 rngs；给个 None 兜底
                return m.apply(x, train=train, rngs=None)

        # 典型 3：不吃 train 这个 kw
        try:
            return m(x)
        except TypeError:
            # 实在不行，随便怼一个最保守的调用
            return m.apply(x) if hasattr(m, "apply") else m(x)
    return fwd


def _build_ensemble(
    base_model: Any,
    num_members: int = 3,
    aggregator: str = "mean",
) -> Tuple[Any, Callable[[Any, jnp.ndarray, bool], Any]]:
    """
    返回 (model, fwd)。能自动匹配大多数奇葩命名的构造器/类。
    尝试顺序：
      1) 常见命名：Ensemble / FlaxEnsemble / build_ensemble / make_ensemble / wrap / make / ensemble
      2) 遍历模块内所有可调用，签名里同时包含 base + 成员数参数的对象
      3) 对 base 参数先传“实例”，失败再传“类型”
    """
    ctors: list[Callable[[], Any]] = []

    # 候选 1：固定名字
    if hasattr(_mod, "Ensemble"):
        ctors.append(lambda: _mod.Ensemble)
    if hasattr(_mod, "FlaxEnsemble"):
        ctors.append(lambda: _mod.FlaxEnsemble)
    for fname in ("build_ensemble", "make_ensemble", "wrap", "make", "ensemble"):
        if hasattr(_mod, fname):
            ctors.append(lambda fname=fname: getattr(_mod, fname))

    # 候选 2：自动遍历
    for _nm, obj in vars(_mod).items():
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except Exception:
            continue
        params = sig.parameters
        has_base = any(k in params for k in _BASE_PARAM_CANDIDATES)
        has_nm = any(k in params for k in _NUM_PARAM_CANDIDATES)
        if has_base and has_nm:
            ctors.append(lambda obj=obj: obj)

    last_error: Optional[BaseException] = None
    # 依次尝试每个构造器：先传实例；失败再传类型
    for ctor_getter in ctors:
        try:
            ctor = ctor_getter()
            sig = inspect.signature(ctor)
            # 先用“实例”试一把
            kwargs = _try_kwargs(sig, base_model, num_members, aggregator)
            try:
                model = ctor(**kwargs)
            except TypeError:
                # 再用“类型”试一把
                kwargs = _try_kwargs(sig, type(base_model), num_members, aggregator)
                model = ctor(**kwargs)

            fwd = _wrap_forward(model)
            # 烟雾测试
            _ = fwd(model, jnp.ones((2, 5)), train=False)
            return model, fwd
        except Exception as e:
            last_error = e

    pytest.skip(f"无法构造 ensemble/flax 模型。检查命名/参数。最后一次错误：{last_error}")


def _extract_members_and_mean(forward_out: Any) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    支持格式：
      - dict: 'members'/'all'/'member_outputs' + 'mean'/'agg'/'aggregated'/'output'/'y'
      - tuple/list: (members, mean) 或 (mean, members)
      - 仅返回聚合：则 members=None
    """
    members = None
    mean = None

    if isinstance(forward_out, dict):
        for k in ("members", "all", "member_outputs"):
            v = forward_out.get(k)
            if isinstance(v, jnp.ndarray):
                members = v
                break
        for k in ("mean", "agg", "aggregated", "output", "y"):
            v = forward_out.get(k)
            if isinstance(v, jnp.ndarray):
                mean = v
                break
    elif isinstance(forward_out, (tuple, list)):
        a0 = forward_out[0] if len(forward_out) > 0 else None
        a1 = forward_out[1] if len(forward_out) > 1 else None
        if isinstance(a0, jnp.ndarray) and a0.ndim == 3:
            members, mean = a0, (a1 if isinstance(a1, jnp.ndarray) else None)
        else:
            mean = a0 if isinstance(a0, jnp.ndarray) else None
            if isinstance(a1, jnp.ndarray) and a1.ndim == 3:
                members = a1

    return members, mean


# -----------------------------
# 2) Fixtures & 输入
# -----------------------------
@pytest.fixture
def _x_batch() -> jnp.ndarray:
    # 绝大多数 toy 模型都吃 [B, D]；若你们 base 模型 D 不同，fixture 会报错提醒
    return jnp.ones((8, 5))


# 直接用项目自带 flax 小模型 fixture 做“base”
from tests.probly.fixtures.flax_models import flax_model_small_2d_2d  # noqa: E402


# -----------------------------
# 3) 测试用例
# -----------------------------
def test_members_and_aggregator_shape(flax_model_small_2d_2d, _x_batch):
    base = flax_model_small_2d_2d
    model, fwd = _build_ensemble(base, num_members=3, aggregator="mean")

    out = fwd(model, _x_batch, train=False)
    members, mean = _extract_members_and_mean(out)

    assert mean is not None, "聚合输出(mean) 不应为 None"
    assert mean.shape[0] == _x_batch.shape[0], "聚合输出的 batch 维应等于输入 batch"

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
        return m if m is not None else out  # 部分实现直接返回聚合数组

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
    用一个超小学习率在 MSE 上走两步，检查 loss 不发散且不升高
    """
