import pytest
import probly.transformation.evidential.regression as regression


def test_evidential_regression_import():
    assert hasattr(regression, "evidential_regression")
    assert callable(regression.evidential_regression)


def test_register_import():
    assert hasattr(regression, "register")
    assert callable(regression.register)


def test_torch_register_function():
    assert hasattr(regression, "_")
