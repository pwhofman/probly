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


def generate_flax_ensemble(model: Any, n_members: int):
    if isinstance(model, type) and issubclass(model, nn.Module):
        base_module = model
        base_kwargs: Dict[str, Any] | None = None
    elif isinstance(model, nn.Module):
        base_module = model.__class__
        base_kwargs = {}
        if is_dataclass(model):
            try:
                # Take only public dataclass fields get rid of internal ones(the _ starting)
                all_fields = asdict(model)
                base_kwargs = {k: v for k, v in all_fields.items() if not k.startswith("_") and k not in ("name", "parent")}
                # If no kwargs remain sets to None
                if not base_kwargs:
                    base_kwargs = None
            # If Error occurs during extraction fall back to set to None
            except Exception:
                base_kwargs = None
    else:
        raise TypeError("generate_flax_ensemble expects a Flax Module class or instance")

    return FlaxEnsemble(base_module=base_module, base_kwargs=base_kwargs, n_members=n_members)


register(nn.Module, generate_flax_ensemble)
