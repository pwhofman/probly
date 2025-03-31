"""Markers for optional dependencies.

Markers can be imported in the test files to skip tests if certain packages are not installed.
Note that we want to avoid marked tests as best as possible, but sometimes dependencies are lagging
behind the latest version of the package or the python version.

Markers should only be used on optional dependencies, i.e. packages that are not required to run
`probly`.

"""

import importlib.util

import pytest

__all__ = [
    "skip_if_no_keras",
    "skip_if_no_lightgbm",
    "skip_if_no_sklearn",
    "skip_if_no_tabpfn",
    "skip_if_no_tensorflow",
    "skip_if_no_torchvision",
    "skip_if_no_xgboost",
]


def is_installed(pkg_name: str) -> bool:
    """Check if a package is installed without importing it."""
    return importlib.util.find_spec(pkg_name) is not None


# torch related: (torch itself is a main dependency) -----------------------------------------------
skip_if_no_torchvision = pytest.mark.skipif(
    not is_installed("torchvision"), reason="torchvision is not installed"
)

# sklearn-like: ------------------------------------------------------------------------------------
skip_if_no_sklearn = pytest.mark.skipif(
    not is_installed("sklearn"), reason="sklearn is not installed"
)

skip_if_no_xgboost = pytest.mark.skipif(
    not is_installed("xgboost"), reason="xgboost is not installed"
)

skip_if_no_lightgbm = pytest.mark.skipif(
    not is_installed("lightgbm"), reason="lightgbm is not installed"
)

# tensorflow related: ------------------------------------------------------------------------------
skip_if_no_tensorflow = pytest.mark.skipif(
    not is_installed("tensorflow"), reason="tensorflow is not installed"
)

skip_if_no_keras = pytest.mark.skipif(not is_installed("keras"), reason="keras is not installed")

# misc: --------------------------------------------------------------------------------------------
skip_if_no_tabpfn = pytest.mark.skipif(
    not is_installed("tabpfn"), reason="TabPFN is not available."
)
