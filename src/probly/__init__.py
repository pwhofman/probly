"""probly: Uncertainty Representation and Quantification for Machine Learning."""

try:
    from probly._version import __version__ as __version__
except ImportError:  # pragma: no cover - only hit in a source tree that was never built
    # setuptools-scm writes _version.py at build/install time; a bare checkout has none.
    __version__ = "0.0.0+unknown"

from probly import (
    datasets as datasets,
    decider as decider,
    evaluation as evaluation,
    integrations as integrations,
    layers as layers,
    method as method,
    metrics as metrics,
    plot as plot,
    predictor as predictor,
    quantification as quantification,
    representation as representation,
    representer as representer,
    train as train,
    transformation as transformation,
    traverse_nn as traverse_nn,
    utils as utils,
)
