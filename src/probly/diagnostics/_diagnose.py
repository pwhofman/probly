"""Minimal diagnostics for trained uncertainty methods.

The entry point is :func:`diagnose`. It takes a fitted representer (e.g.
``representer(model, num_samples=50)``), test inputs and targets, and optionally
out-of-distribution inputs and a deterministic baseline predictor. Each
diagnostic yields a pass/warn/fail/skip verdict; verdicts are baseline-relative
where possible, since absolute thresholds conflate "method is broken" with
"task does not elicit this property".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from probly.decider import categorical_from_mean
from probly.evaluation.ood import out_of_distribution_detection_auroc
from probly.evaluation.tasks import selective_prediction
from probly.metrics import accuracy, expected_calibration_error
from probly.predictor import predict
from probly.quantification import quantify

from ._report import DiagnosticReport, DiagnosticResult, Verdict

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representer import Representer

_ADDITIVITY_RTOL = 1e-3
_AURC_PASS_RATIO = 0.95
_BASELINE_WARN_RATIO = 1.05
_BASELINE_FAIL_RATIO = 1.25
_AURC_ABS_TOL = 1e-4
_OOD_PASS_AUROC = 0.6
_OOD_FAIL_AUROC = 0.4
_SELECTIVE_PREDICTION_BINS = 50
_CONSTANT_RTOL = 1e-6


def _skip(name: str, detail: str) -> DiagnosticResult:
    return DiagnosticResult(name=name, verdict=Verdict.SKIP, detail=detail)


def _to_numpy(values: Any) -> np.ndarray:  # noqa: ANN401
    """Convert an array-like (including torch tensors) to a numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


def _aurc(criterion: np.ndarray, losses: np.ndarray) -> float:
    """Binned AURC from the selective-prediction task, with bins capped by the sample count."""
    aurc, _ = selective_prediction(criterion, losses, n_bins=min(_SELECTIVE_PREDICTION_BINS, len(losses)))
    return aurc


def _as_probabilities(output: Any) -> np.ndarray:  # noqa: ANN401
    """Interpret a predictor output as class probabilities, applying softmax to logits."""
    if hasattr(output, "probabilities"):
        return _to_numpy(output.probabilities)
    values = _to_numpy(output)
    if np.all(values >= 0.0) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-4):
        return values
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _check_decomposition(decomposition: Any) -> DiagnosticResult:  # noqa: ANN401
    name = "decomposition_additivity"
    names = {notion.__name__ for notion in decomposition.components}
    if not {"TotalUncertainty", "AleatoricUncertainty", "EpistemicUncertainty"} <= names:
        return _skip(name, "no total/aleatoric/epistemic decomposition")
    total = _to_numpy(decomposition["total"])
    residual = float(
        np.abs(total - (_to_numpy(decomposition["aleatoric"]) + _to_numpy(decomposition["epistemic"]))).max()
    )
    scale = max(float(np.abs(total).max()), 1e-12)
    verdict = Verdict.PASS if residual <= _ADDITIVITY_RTOL * scale else Verdict.WARN
    return DiagnosticResult(name=name, verdict=verdict, value=residual, detail="max |total - (aleatoric + epistemic)|")


def _check_uncertainty_variation(uncertainty: np.ndarray) -> DiagnosticResult:
    name = "uncertainty_variation"
    spread = float(uncertainty.max() - uncertainty.min())
    scale = max(float(np.abs(uncertainty).max()), 1e-12)
    relative = spread / scale
    verdict = Verdict.PASS if relative > _CONSTANT_RTOL else Verdict.FAIL
    return DiagnosticResult(
        name=name,
        verdict=verdict,
        value=relative,
        reference=_CONSTANT_RTOL,
        detail="peak-to-peak of total uncertainty relative to its scale",
    )


def _check_selective_prediction(uncertainty: np.ndarray, errors: np.ndarray) -> DiagnosticResult:
    name = "selective_prediction"
    random_aurc = float(errors.mean())
    if random_aurc == 0.0:
        return _skip(name, "no errors on the test data")
    aurc = _aurc(uncertainty, errors)
    if aurc < _AURC_PASS_RATIO * random_aurc:
        verdict = Verdict.PASS
    elif aurc < random_aurc:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    return DiagnosticResult(
        name=name, verdict=verdict, value=aurc, reference=random_aurc, detail="AURC vs random rejection"
    )


def _check_ood(epistemic_id: np.ndarray, epistemic_ood: np.ndarray) -> DiagnosticResult:
    name = "ood_separation"
    value = out_of_distribution_detection_auroc(epistemic_id, epistemic_ood)
    if value > _OOD_PASS_AUROC:
        verdict = Verdict.PASS
    elif value >= _OOD_FAIL_AUROC:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    return DiagnosticResult(
        name=name, verdict=verdict, value=value, reference=0.5, detail="AUROC of epistemic uncertainty"
    )


def _ood_result(
    method: Representer[Any, Any, Any, Any],
    decomposition: Any,  # noqa: ANN401
    ood_inputs: Any,  # noqa: ANN401
) -> DiagnosticResult:
    if ood_inputs is None:
        return _skip("ood_separation", "no ood_inputs given")
    if not any(notion.__name__ == "EpistemicUncertainty" for notion in decomposition.components):
        return _skip("ood_separation", "no epistemic uncertainty available")
    try:
        epistemic = _to_numpy(decomposition["epistemic"])
        epistemic_ood = _to_numpy(cast("Any", quantify(method.represent(ood_inputs)))["epistemic"])
    except Exception as error:  # noqa: BLE001
        return _skip("ood_separation", f"{type(error).__name__}: {error}")
    return _check_ood(epistemic, epistemic_ood)


def _check_against_baseline(
    baseline_probabilities: np.ndarray,
    targets: np.ndarray,
    method_aurc: float,
) -> DiagnosticResult:
    name = "baseline_selective_prediction"
    probabilities = baseline_probabilities
    errors = (probabilities.argmax(axis=-1) != targets).astype(float)
    if errors.mean() == 0.0:
        return _skip(name, "baseline makes no errors on the test data")
    baseline_aurc = _aurc(1.0 - probabilities.max(axis=-1), errors)
    if method_aurc <= baseline_aurc * _BASELINE_WARN_RATIO + _AURC_ABS_TOL:
        verdict = Verdict.PASS
    elif method_aurc <= baseline_aurc * _BASELINE_FAIL_RATIO + _AURC_ABS_TOL:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    return DiagnosticResult(
        name=name, verdict=verdict, value=method_aurc, reference=baseline_aurc, detail="AURC vs max-softmax baseline"
    )


def diagnose(
    method: Representer[Any, Any, Any, Any],
    inputs: Any,  # noqa: ANN401
    targets: Any,  # noqa: ANN401
    ood_inputs: Any = None,  # noqa: ANN401
    baseline: Predictor | None = None,
) -> DiagnosticReport:
    """Run the minimal diagnostic suite on a fitted representer.

    Args:
        method: Fitted representer of the uncertainty method under test.
        inputs: Test inputs.
        targets: Integer class targets for the test inputs.
        ood_inputs: Optional out-of-distribution inputs.
        baseline: Optional deterministic predictor compared on selective prediction.

    Returns:
        The diagnostic report.
    """
    results: list[DiagnosticResult] = []
    downstream = [
        "uncertainty_variation",
        "accuracy",
        "ece",
        "decomposition_additivity",
        "selective_prediction",
        "ood_separation",
        "baseline_selective_prediction",
    ]

    try:
        representation = method.represent(inputs)
        decomposition = cast("Any", quantify(representation))
        uncertainty = _to_numpy(decomposition["total"])
    except Exception as error:  # noqa: BLE001
        results.append(
            DiagnosticResult(name="pipeline", verdict=Verdict.FAIL, detail=f"{type(error).__name__}: {error}")
        )
        results += [_skip(name, "pipeline failed") for name in downstream]
        return DiagnosticReport(tuple(results))
    results.append(DiagnosticResult(name="pipeline", verdict=Verdict.PASS, detail="represent and quantify succeeded"))
    results.append(_check_uncertainty_variation(uncertainty))

    targets_np = _to_numpy(targets)
    baseline_probabilities = None if baseline is None else _as_probabilities(predict(baseline, inputs))

    method_aurc = None
    try:
        categorical = categorical_from_mean(representation)
    except NotImplementedError:
        results += [_skip(name, "no categorical decision available") for name in ("accuracy", "ece")]
        results.append(_check_decomposition(decomposition))
        results.append(_skip("selective_prediction", "no categorical decision available"))
    else:
        probabilities = _to_numpy(categorical.probabilities)
        errors = (probabilities.argmax(axis=-1) != targets_np).astype(float)
        baseline_accuracy = baseline_ece = None
        if baseline_probabilities is not None:
            baseline_accuracy = float(accuracy(baseline_probabilities, targets_np))  # ty:ignore[invalid-argument-type]
            baseline_ece = float(expected_calibration_error(baseline_probabilities, targets_np))  # ty:ignore[invalid-argument-type]
        results.append(
            DiagnosticResult(
                name="accuracy",
                verdict=Verdict.INFO,
                value=float(accuracy(categorical, targets)),  # ty:ignore[invalid-argument-type]
                reference=baseline_accuracy,
                detail="accuracy of the mean prediction (reference: baseline)",
            )
        )
        results.append(
            DiagnosticResult(
                name="ece",
                verdict=Verdict.INFO,
                value=float(expected_calibration_error(categorical, targets)),  # ty:ignore[invalid-argument-type]
                reference=baseline_ece,
                detail="expected calibration error of the mean prediction (reference: baseline)",
            )
        )
        results.append(_check_decomposition(decomposition))
        selective = _check_selective_prediction(uncertainty, errors)
        method_aurc = selective.value
        results.append(selective)

    results.append(_ood_result(method, decomposition, ood_inputs))

    if baseline_probabilities is None:
        results.append(_skip("baseline_selective_prediction", "no baseline given"))
    elif method_aurc is None:
        results.append(_skip("baseline_selective_prediction", "method AURC unavailable"))
    else:
        results.append(_check_against_baseline(baseline_probabilities, targets_np, method_aurc))

    return DiagnosticReport(tuple(results))
