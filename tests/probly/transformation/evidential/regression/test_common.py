from __future__ import annotations
from typing import Any, Callable, Dict, Type
import copy

REPLACED_LAST_LINEAR = "replaced_last_linear"
_REGISTRY: Dict[Type, Callable[[Any, Dict[str, Any]], Any]] = {}

def register(*, cls: Type, traverser: Callable[[Any, Dict[str, Any]], Any]) -> None:
    _REGISTRY[cls] = traverser

def evidential_regression(model: Any) -> Any:
    clone = copy.deepcopy(model)
    state: Dict[str, Any] = {REPLACED_LAST_LINEAR: False}
    return _transform(clone, state)

def _transform(node: Any, state: Dict[str, Any]) -> Any:
    if isinstance(node, (list, tuple)):
        seq = list(node)
        for i in reversed(range(len(seq))):
            seq[i] = _transform(seq[i], state)
        return type(node)(seq)
    if isinstance(node, dict):
        for k in list(node.keys())[::-1]:
            node[k] = _transform(node[k], state)
        return node
    try:
        attrs = list(vars(node).items())
    except TypeError:
        attrs = []
    for name, value in attrs[::-1]:
        new_val = _transform(value, state)
        if new_val is not value:
            setattr(node, name, new_val)
    for cls, traverser in _REGISTRY.items():
        if isinstance(node, cls):
            return traverser(node, state)
    return node

__all__ = ["register", "evidential_regression", "REPLACED_LAST_LINEAR"]


