"""Gemma 4 experiment: semantic entropy, spectral uncertainty, and calibration utilities."""

MODEL_ID = "google/gemma-4-E2B-it"

from gemma_experiment.calibration import (
    CALIBRATORS,
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
    average_calibration_error,
    compute_aggregates,
    compute_semantic_confidence_discrete,
    compute_semantic_confidence_weighted,
    expected_calibration_error,
    leave_one_out_evaluate,
    reliability_diagram_data,
)
from gemma_experiment.clarification import generate_clarifications
from gemma_experiment.correctness import check_cluster_correctness
from gemma_experiment.embeddings import DEFAULT_EMBED_MODEL, EmbedModel, SentenceEmbedder
from gemma_experiment.entailment import DEFAULT_NLI_MODEL, EntailmentModel, NLIModel
from gemma_experiment.generation import generate_responses
from gemma_experiment.paths import CACHE_DIR, DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from gemma_experiment.semantic_entropy import (
    cluster_assignment_entropy,
    get_semantic_ids,
    weighted_semantic_entropy,
)
from gemma_experiment.setup import suppress_hf_noise
from gemma_experiment.spectral_uncertainty import (
    SpectralDecomposition,
    rbf_kernel_matrix,
    spectral_decomposed_uncertainty,
    spectral_total_uncertainty,
    von_neumann_entropy,
)

__all__ = [
    "CACHE_DIR",
    "CALIBRATORS",
    "DATA_DIR",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_NLI_MODEL",
    "MODEL_ID",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "EmbedModel",
    "EntailmentModel",
    "IsotonicCalibrator",
    "NLIModel",
    "PlattScaler",
    "SentenceEmbedder",
    "SpectralDecomposition",
    "TemperatureScaler",
    "average_calibration_error",
    "check_cluster_correctness",
    "cluster_assignment_entropy",
    "compute_aggregates",
    "compute_semantic_confidence_discrete",
    "compute_semantic_confidence_weighted",
    "expected_calibration_error",
    "generate_clarifications",
    "generate_responses",
    "get_semantic_ids",
    "leave_one_out_evaluate",
    "rbf_kernel_matrix",
    "reliability_diagram_data",
    "spectral_decomposed_uncertainty",
    "spectral_total_uncertainty",
    "suppress_hf_noise",
    "von_neumann_entropy",
    "weighted_semantic_entropy",
]
