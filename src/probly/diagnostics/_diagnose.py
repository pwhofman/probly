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
from probly.predictor import predict
from probly.quantification import quantify

from ._metrics import area_under_risk_coverage, auroc, to_numpy
from ._report import DiagnosticReport, DiagnosticResult, Verdict

if TYPE_CHECKING:
    from probly.predictor import Predictor
    from probly.representer import Representer

_ADDITIVITY_RTOL = 1e-3
_AURC_PASS_RATIO = 0.95
_BASELINE_WARN_RATIO = 1.05
_BASELINE_FAIL_RATIO = 1.25
_OOD_PASS_AUROC = 0.6
_OOD_FAIL_AUROC = 0.4


def _skip(name: str, detail: str) -> DiagnosticResult:
    return DiagnosticResult(name=name, verdict=Verdict.SKIP, detail=detail)


def _as_probabilities(output: Any) -> np.ndarray:  # noqa: ANN401
    """Interpret a predictor output as class probabilities, applying softmax to logits."""
    if hasattr(output, "probabilities"):
        return to_numpy(output.probabilities)
    values = to_numpy(output)
    if np.all(values >= 0.0) and np.allclose(values.sum(axis=-1), 1.0, atol=1e-4):
        return values
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _check_decomposition(decomposition: Any) -> DiagnosticResult:  # noqa: ANN401
    name = "decomposition_additivity"
    names = {notion.__name__ for notion in decomposition.components}
    if not {"TotalUncertainty", "AleatoricUncertainty", "EpistemicUncertainty"} <= names:
        return _skip(name, "no total/aleatoric/epistemic decomposition")
    total = to_numpy(decomposition["total"])
    residual = float(
        np.abs(total - (to_numpy(decomposition["aleatoric"]) + to_numpy(decomposition["epistemic"]))).max()
    )
    scale = max(float(np.abs(total).max()), 1e-12)
    verdict = Verdict.PASS if residual <= _ADDITIVITY_RTOL * scale else Verdict.WARN
    return DiagnosticResult(name=name, verdict=verdict, value=residual, detail="max |total - (aleatoric + epistemic)|")


def _check_selective_prediction(uncertainty: np.ndarray, errors: np.ndarray) -> DiagnosticResult:
    name = "selective_prediction"
    random_aurc = float(errors.mean())
    if random_aurc == 0.0:
        return _skip(name, "no errors on the test data")
    aurc = area_under_risk_coverage(uncertainty, errors)
    if aurc < _AURC_PASS_RATIO * random_aurc:
        verdict = Verdict.PASS
    elif aurc < random_aurc:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    return DiagnosticResult(
        name=name, verdict=verdict, value=aurc, reference=random_aurc, detail="AURC vs random rejection"
    )


def _check_ood(uncertainty_id: np.ndarray, uncertainty_ood: np.ndarray) -> DiagnosticResult:
    name = "ood_separation"
    value = auroc(uncertainty_id, uncertainty_ood)
    if value > _OOD_PASS_AUROC:
        verdict = Verdict.PASS
    elif value >= _OOD_FAIL_AUROC:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.FAIL
    return DiagnosticResult(name=name, verdict=verdict, value=value, reference=0.5, detail="AUROC of total uncertainty")


def _check_against_baseline(
    baseline: Predictor,
    inputs: Any,  # noqa: ANN401
    targets: np.ndarray,
    method_aurc: float,
) -> DiagnosticResult:
    name = "baseline_selective_prediction"
    probabilities = _as_probabilities(predict(baseline, inputs))
    errors = (probabilities.argmax(axis=-1) != targets).astype(float)
    if errors.mean() == 0.0:
        return _skip(name, "baseline makes no errors on the test data")
    baseline_aurc = area_under_risk_coverage(1.0 - probabilities.max(axis=-1), errors)
    ratio = method_aurc / max(baseline_aurc, 1e-12)
    if ratio <= _BASELINE_WARN_RATIO:
        verdict = Verdict.PASS
    elif ratio <= _BASELINE_FAIL_RATIO:
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
    downstream = ["decomposition_additivity", "selective_prediction", "ood_separation", "baseline_selective_prediction"]

    try:
        representation = method.represent(inputs)
        decomposition = cast("Any", quantify(representation))
        uncertainty = to_numpy(decomposition["total"])
    except Exception as error:  # noqa: BLE001
        results.append(
            DiagnosticResult(name="pipeline", verdict=Verdict.FAIL, detail=f"{type(error).__name__}: {error}")
        )
        results += [_skip(name, "pipeline failed") for name in downstream]
        return DiagnosticReport(tuple(results))
    results.append(DiagnosticResult(name="pipeline", verdict=Verdict.PASS, detail="represent and quantify succeeded"))

    results.append(_check_decomposition(decomposition))

    targets = to_numpy(targets)
    method_aurc = None
    try:
        probabilities = to_numpy(categorical_from_mean(representation).probabilities)
    except NotImplementedError:
        results.append(_skip("selective_prediction", "no categorical decision available"))
    else:
        errors = (probabilities.argmax(axis=-1) != targets).astype(float)
        selective = _check_selective_prediction(uncertainty, errors)
        method_aurc = selective.value
        results.append(selective)

    if ood_inputs is None:
        results.append(_skip("ood_separation", "no ood_inputs given"))
    else:
        try:
            uncertainty_ood = to_numpy(cast("Any", quantify(method.represent(ood_inputs)))["total"])
            results.append(_check_ood(uncertainty, uncertainty_ood))
        except Exception as error:  # noqa: BLE001
            results.append(_skip("ood_separation", f"{type(error).__name__}: {error}"))

    if baseline is None:
        results.append(_skip("baseline_selective_prediction", "no baseline given"))
    elif method_aurc is None:
        results.append(_skip("baseline_selective_prediction", "method AURC unavailable"))
    else:
        results.append(_check_against_baseline(baseline, inputs, targets, method_aurc))

    return DiagnosticReport(tuple(results))
