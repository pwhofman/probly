"""Ensemble Flax implementation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp

from .common import register


class FlaxEnsemble(nn.Module):
    """FlaxEnsemble class."""

    base_module: type
    base_kwargs: dict[str, Any] | None = None
    n_members: int = 10

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        return_all: bool = False,
        **call_kwargs: object,
    ) -> jnp.ndarray | dict[str, jnp.ndarray]:
        """Apply each ensemble member and aggregate."""
        outputs: list[object] = []
        for i in range(self.n_members):
            ctor_kwargs = self.base_kwargs or {}
            member = self.base_module(**ctor_kwargs, name=f"member_{i}")
            y = member(x, **call_kwargs)
            outputs.append(y)

        stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=1), *outputs)
        if return_all:
            return stacked  # type: ignore[return-value]
        averaged = jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=1), stacked)
        return averaged  # type: ignore[return-value]


def generate_flax_ensemble(
    model: nn.Module | type[nn.Module] | str,
    n_members: int,
) -> FlaxEnsemble:
    """Create a class:FlaxEnsemble from a module instance or class."""
    base_module = model.__class__ if not isinstance(model, type) else model
    base_kwargs: dict[str, Any] | None = None
    if not isinstance(model, type) and is_dataclass(model):
        try:
            # mypy: after is_dataclass check, model is a dataclass instance
            all_fields = asdict(model)  # type: ignore[arg-type]
            filtered = {k: v for k, v in all_fields.items() if not k.startswith("_") and k not in ("name", "parent")}
            if filtered:
                base_kwargs = filtered
        except (TypeError, ValueError):
            base_kwargs = None
    return FlaxEnsemble(base_module=base_module, base_kwargs=base_kwargs, n_members=n_members)


register(nn.Module, generate_flax_ensemble)
