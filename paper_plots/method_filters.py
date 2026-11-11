"""Per-method client-side filters applied before a run gets cached.

Use when a wandb project has multiple sweeps for the same
``(method, dataset, base_model)`` and you need to pin a specific
hyperparameter combination. Keys are dotted paths into ``run.config``;
values must match exactly. A method without an entry is unfiltered.
"""

from __future__ import annotations

from typing import Any

METHOD_PARAM_FILTERS: dict[str, dict[str, Any]] = {
    # EffCre paper-quality SP/OOD numbers use alpha=0.8 (config default is 0.5).
    "efficient_credal_prediction": {"method.train.alpha": 0.8},
}


def nested_get(cfg: Any, dotted: str) -> Any:  # noqa: ANN401
    """Walk ``dotted`` into ``cfg`` (dict-of-dicts). Returns ``None`` on miss."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def params_match(cfg: Any, params_check: dict[str, Any]) -> bool:  # noqa: ANN401
    """``True`` iff every entry in ``params_check`` equals the value at that path in ``cfg``."""
    return all(nested_get(cfg, dotted) == expected for dotted, expected in params_check.items())


def passes(method: str, cfg: Any) -> bool:  # noqa: ANN401
    """``True`` iff ``cfg`` satisfies the registered filter for ``method`` (or none is set)."""
    return params_match(cfg, METHOD_PARAM_FILTERS.get(method, {}))
