"""All errors and warnings used in the probly package."""

from __future__ import annotations

from warnings import warn


def raise_deprecation_warning(
    message: str,
    deprecated_in: str,
    removed_in: str,
) -> None:
    """Raise a deprecation warning with the given details."""
    message += f" This feature is deprecated in version {deprecated_in} and will be removed in version {removed_in}."
    warn(message, DeprecationWarning, stacklevel=2)
