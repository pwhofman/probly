"""Init module for evaluation implementations."""

import importlib
from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    if name == "active_learning":
        module = importlib.import_module(f"{__name__}.active_learning")
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
