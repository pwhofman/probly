"""river-uncertainty: experimental bridge between river online learners and probly uncertainty tooling."""

from river_uncertainty.experiment import (
    PrequentialTrace,
    detect_drift,
    first_arf_drift_after,
    run_prequential,
    run_prequential_generic,
)
from river_uncertainty.paths import RESULTS_DIR
from river_uncertainty.plotting import rolling_mean
from river_uncertainty.representation import (
    ARFEnsembleRepresentation,
    river_arf_to_probly_sample,
)
from river_uncertainty.stream import make_synthetic_stream

__all__ = [
    "RESULTS_DIR",
    "ARFEnsembleRepresentation",
    "PrequentialTrace",
    "detect_drift",
    "first_arf_drift_after",
    "make_synthetic_stream",
    "river_arf_to_probly_sample",
    "rolling_mean",
    "run_prequential",
    "run_prequential_generic",
]

try:
    from river_uncertainty.deep_classifier import OnlineClassifier
    from river_uncertainty.deep_networks import DropoutMLP
    from river_uncertainty.deep_representation import (
        DeepRepresentation,
        deep_ensemble_to_probly_sample,
        mc_dropout_to_probly_sample,
    )

    __all__ += [
        "DeepRepresentation",
        "DropoutMLP",
        "OnlineClassifier",
        "deep_ensemble_to_probly_sample",
        "mc_dropout_to_probly_sample",
    ]
except ModuleNotFoundError:
    pass  # torch not installed; deep-learning extras unavailable
