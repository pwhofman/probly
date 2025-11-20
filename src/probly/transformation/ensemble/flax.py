from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Type, Dict
from dataclasses import asdict, is_dataclass

from .common import register


class FlaxEnsemble(nn.Module):
    base_module: Type[nn.Module]
    base_kwargs: Dict[str, Any] | None = None
    n_members: int = 10

    @nn.compact
    def __call__(self, x, *, return_all: bool = False, **call_kwargs):
        outputs = []
        for i in range(self.n_members):
            ctor_kwargs = self.base_kwargs or {}
            member = self.base_module(**ctor_kwargs, name=f"member_{i}")
            y = member(x, **call_kwargs)
            outputs.append(y)

        # Support arbitrary PyTree outputs by stacking/averaging per-leaf
        stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=1), *outputs)
        if return_all:
            return stacked
        averaged = jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=1), stacked)
        return averaged


def generate_flax_ensemble(model: nn.Module, n_members: int):
    base_module = model.__class__
    base_kwargs: Dict[str, Any] | None = None
    if is_dataclass(model):
        try:
            all_fields = asdict(model)
            # Filter out internal and framework-managed fields
            filtered = {k: v for k, v in all_fields.items() if not k.startswith("_") and k not in ("name", "parent")}
            if filtered:
                base_kwargs = filtered
        except Exception:
            base_kwargs = None
    return FlaxEnsemble(base_module=base_module, base_kwargs=base_kwargs, n_members=n_members)


register(nn.Module, generate_flax_ensemble)
