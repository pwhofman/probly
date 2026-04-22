"""Lazy alternatives to singledispatch and isinstance."""

from lazy_dispatch.isinstance import LazyType, lazy_isinstance, lazy_issubclass
from lazy_dispatch.load import lazy_callable, lazy_import
from lazy_dispatch.registry_meta import (
    ProtocolRegistry,
    ProtocolRegistryMeta,
    Registry,
    RegistryMeta,
    annotator,
)
from lazy_dispatch.singledispatch import Lazydispatch, is_valid_dispatch_type, lazydispatch

from . import registry_pickle

__all__ = [
    "LazyType",
    "Lazydispatch",
    "ProtocolRegistry",
    "ProtocolRegistryMeta",
    "Registry",
    "RegistryMeta",
    "annotator",
    "is_valid_dispatch_type",
    "lazy_callable",
    "lazy_import",
    "lazy_isinstance",
    "lazy_issubclass",
    "lazydispatch",
    "registry_pickle",
]
