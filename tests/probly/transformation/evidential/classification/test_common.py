"""Test for evidential classification models - common tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from probly.predictor import Predictor
from probly.transformation.evidential.classification import evidential_classification


class TestBasicFunctionality:
    """Test class for basic evidential classification functionality."""

    def test_unregistered_type_raises_error(self, dummy_predictor: Predictor) -> None:
        """Tests that evidential_classification raises error for unregistered types.

        This function verifies that the evidential_classification transformation
        raises NotImplementedError for predictor types that haven't been registered.

        Parameters:
            dummy_predictor: A dummy predictor to be tested.

        Raises:
            AssertionError: If NotImplementedError is not raised.
        """
        with pytest.raises(NotImplementedError, match="No evidential classification appender registered"):
            evidential_classification(dummy_predictor)

    def test_with_none_input(self) -> None:
        """Tests that evidential_classification raises appropriate error with None input.

        This function verifies that passing None to evidential_classification
        raises NotImplementedError.

        Raises:
            AssertionError: If the function does not raise an error with None input.
        """
        with pytest.raises(NotImplementedError):
            evidential_classification(cast(Predictor[Any, Any, Any], None))
