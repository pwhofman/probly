"""Collects all deprecated behaviour tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.utils.errors import raise_deprecation_warning

if TYPE_CHECKING:
    import pytest

from .utils import register_deprecated


@register_deprecated(
    name="ExampleFeature(NeverRemoveThis)",
    deprecated_in="1.0.0",
    removed_in="9.9.9",
)
def example_always_warn(requests: pytest.FixtureRequest) -> None:
    """An example feature that always raises a deprecation warning using a fixture.

    This "behavior" is just an example of how to handle deprecations in the codebase. This
    "feature" needs a fixture (in this case an example game) which is passed via the `request`
    fixture. The warning which is raised contains the same version information as the
    `DeprecatedFeature` instance in the `DEPRECATED_FEATURES` list.

    Note:
        Do not delete this "feature", as it is a good example of how to handle deprecations and
        write your own deprecation tests.

    """
    raise_deprecation_warning(
        "This is an example of a deprecated feature that always raises a warning.",
        deprecated_in="1.0.0",
        removed_in="9.9.9",
    )
    _ = requests.getfixturevalue("conv_linear_model")  # gets a fixture as an example of usage
