"""Lazy alternatives to singledispatch and isinstance."""

from . import registry_pickle
from .isinstance import LazyType, lazy_isinstance, lazy_issubclass
from .load import lazy_callable, lazy_import
from .registry_meta import (
    ProtocolRegistry,
    ProtocolRegistryMeta,
    Registry,
    RegistryMeta,
    annotator,
)
from .singledispatch import Flexdispatch, flexdispatch, is_valid_dispatch_type

__all__ = [
    "Flexdispatch",
    "LazyType",
    "ProtocolRegistry",
    "ProtocolRegistryMeta",
    "Registry",
    "RegistryMeta",
    "annotator",
    "flexdispatch",
    "is_valid_dispatch_type",
    "lazy_callable",
    "lazy_import",
    "lazy_isinstance",
    "lazy_issubclass",
    "registry_pickle",
]
