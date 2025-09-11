"""Lazy alternatives to singledispatch and isinstance."""

from lazy_dispatch.isinstance import LazyType, lazy_isinstance, lazy_issubclass
from lazy_dispatch.singledispatch import is_valid_dispatch_type, lazy_singledispatch

__all__ = ["LazyType", "is_valid_dispatch_type", "lazy_isinstance", "lazy_issubclass", "lazy_singledispatch"]
