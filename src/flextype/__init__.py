"""Lazy alternatives to singledispatch and isinstance."""

from . import registry_pickle
from .isinstance import LazyType, lazy_isinstance, lazy_issubclass
from .load import lazy_callable, lazy_import
from .registry_meta import (
    ProtocolRegistry,
    ProtocolRegistryMeta,
    RegistrationError,
    Registry,
    RegistryMeta,
    annotator,
    copy_explicit_registry_classes,
    get_explicit_registry_classes,
)
from .singledispatch import Flexdispatch, flexdispatch, is_valid_dispatch_type

__all__ = [
    "Flexdispatch",
    "LazyType",
    "ProtocolRegistry",
    "ProtocolRegistryMeta",
    "RegistrationError",
    "Registry",
    "RegistryMeta",
    "annotator",
    "copy_explicit_registry_classes",
    "flexdispatch",
    "get_explicit_registry_classes",
    "is_valid_dispatch_type",
    "lazy_callable",
    "lazy_import",
    "lazy_isinstance",
    "lazy_issubclass",
    "registry_pickle",
]
