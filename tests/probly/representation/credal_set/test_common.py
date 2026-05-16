"""Tests for the credal-set common factory fallbacks."""

from __future__ import annotations

import pytest


class TestCredalSetCommonFallbacks:
    """The credal-set factory functions raise for unregistered input types."""

    def test_create_probability_intervals_raises(self) -> None:
        from probly.representation.credal_set._common import create_probability_intervals  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="probability intervals"):
            create_probability_intervals(object())

    def test_create_convex_credal_set_raises(self) -> None:
        from probly.representation.credal_set._common import create_convex_credal_set  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="convex credal"):
            create_convex_credal_set(object())

    def test_create_distance_based_credal_set_raises(self) -> None:
        from probly.representation.credal_set._common import create_distance_based_credal_set  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="distance-based"):
            create_distance_based_credal_set(object())

    def test_create_probability_intervals_from_lower_upper_array_raises(self) -> None:
        from probly.representation.credal_set._common import (  # noqa: PLC0415
            create_probability_intervals_from_lower_upper_array,
        )

        with pytest.raises(NotImplementedError, match="probability intervals"):
            create_probability_intervals_from_lower_upper_array(object())

    def test_create_probability_intervals_from_bounds_raises(self) -> None:
        from probly.representation.credal_set._common import create_probability_intervals_from_bounds  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="probability intervals"):
            create_probability_intervals_from_bounds(object(), object(), object())

    def test_create_distance_based_credal_set_from_center_and_radius_raises(self) -> None:
        from probly.representation.credal_set._common import (  # noqa: PLC0415
            create_distance_based_credal_set_from_center_and_radius,
        )

        with pytest.raises(NotImplementedError, match="distance-based"):
            create_distance_based_credal_set_from_center_and_radius(object(), 0.5)

    def test_create_dirichlet_level_set_credal_set_raises(self) -> None:
        from probly.representation.credal_set._common import create_dirichlet_level_set_credal_set  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Dirichlet"):
            create_dirichlet_level_set_credal_set(object(), 0.5)

    def test_create_mle_probability_intervals_raises(self) -> None:
        from probly.representation.credal_set._common import create_mle_probability_intervals  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="MLE probability intervals"):
            create_mle_probability_intervals(object())

    def test_create_mle_probability_intervals_from_lower_upper_array_raises(self) -> None:
        from probly.representation.credal_set._common import (  # noqa: PLC0415
            create_mle_probability_intervals_from_lower_upper_array,
        )

        with pytest.raises(NotImplementedError, match="MLE probability intervals"):
            create_mle_probability_intervals_from_lower_upper_array(object(), object())
