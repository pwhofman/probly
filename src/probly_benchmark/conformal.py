"""Benchmark split-conformal registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from probly.calibrator import calibrate
from probly.method.calibration import torch_identity_logit_model
from probly.method.conformal import conformal_lac
from probly.predictor import predict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from omegaconf import DictConfig
    import torch


DEFAULT_CONFORMAL = "none"
CONFORMAL_ARTIFACT_TYPE = "split_conformal"
CONFORMAL_STATE_DICT_KEY = "conformal_state_dict"
SOURCE_ARTIFACT_KEY = "source_artifact"
SOURCE_RUN_ID_KEY = "source_run_id"


@dataclass(frozen=True)
class ConformalSpec:
    """Benchmark metadata for a split-conformal method.

    Attributes:
        transform: Function that wraps a predictor with a conformal method.
        supported_methods: Benchmark methods this conformal method is currently allowed for.
        state_keys: Conformal state keys expected in saved conformal artifacts.
        metric_extractors: Functions that extract conformal-method-specific metrics.
    """

    transform: Callable[..., torch.nn.Module]
    supported_methods: frozenset[str]
    state_keys: frozenset[str]
    metric_extractors: tuple[Callable[[torch.nn.Module], dict[str, float]], ...] = ()


def _conformal_lac(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with LAC split-conformal prediction."""
    return cast("torch.nn.Module", conformal_lac(model))


def _quantile_metrics(conformalizer: torch.nn.Module) -> dict[str, float]:
    """Extract the calibrated conformal quantile."""
    quantile = getattr(conformalizer, "conformal_quantile", None)
    if quantile is None:
        return {}
    return {"quantile": float(quantile)}


CONFORMAL_METHODS = {
    "conformal_lac": ConformalSpec(
        transform=_conformal_lac,
        supported_methods=frozenset({"base"}),
        state_keys=frozenset({"_conformal_quantile"}),
        metric_extractors=(_quantile_metrics,),
    ),
}
"""Registry of split-conformal methods supported by the benchmark."""


def get_conformal_name(cfg: DictConfig | dict) -> str:
    """Return the configured conformal method name."""
    conformal = cfg.get("conformal", None)
    if not conformal:
        return DEFAULT_CONFORMAL
    return str(conformal.get("name", DEFAULT_CONFORMAL)).lower()


def validate_conformal_config(cfg: DictConfig | dict, *, allow_none: bool = True) -> None:
    """Validate that the configured conformal method is supported for the benchmark method."""
    name = get_conformal_name(cfg)
    if name == DEFAULT_CONFORMAL:
        if allow_none:
            return
        msg = "conformal=none cannot be used by conformalize.py; choose a split-conformal method."
        raise ValueError(msg)

    spec = CONFORMAL_METHODS.get(name)
    if spec is None:
        supported = ", ".join(sorted(CONFORMAL_METHODS))
        msg = f"Unknown conformal method {name!r}; supported conformal methods: {supported}."
        raise ValueError(msg)

    method_name = str(cfg.get("method", {}).get("name"))
    if method_name not in spec.supported_methods:
        supported_methods = ", ".join(sorted(spec.supported_methods))
        msg = (
            f"conformal.name={name!r} is only supported for benchmark methods "
            f"({supported_methods}); got method={method_name!r}."
        )
        raise ValueError(msg)


def get_conformal_spec(cfg: DictConfig | dict) -> ConformalSpec:
    """Return the configured conformal spec."""
    name = get_conformal_name(cfg)
    if name == DEFAULT_CONFORMAL:
        msg = "conformal=none has no ConformalSpec."
        raise ValueError(msg)
    spec = CONFORMAL_METHODS.get(name)
    if spec is None:
        supported = ", ".join(sorted(CONFORMAL_METHODS))
        msg = f"Unknown conformal method {name!r}; supported conformal methods: {supported}."
        raise ValueError(msg)
    return spec


def conformal_params(cfg: DictConfig | dict) -> dict[str, Any]:
    """Return conformal method constructor parameters as a plain dictionary."""
    conformal_cfg = cfg.get("conformal", {})
    return dict(conformal_cfg.get("params") or {})


def conformal_alpha(cfg: DictConfig | dict) -> float:
    """Return the split-conformal miscoverage level."""
    conformal_cfg = cfg.get("conformal", {})
    if "alpha" not in conformal_cfg:
        msg = "Split-conformal methods require `conformal.alpha`."
        raise ValueError(msg)
    return float(conformal_cfg["alpha"])


def apply_conformal(model: torch.nn.Module, cfg: DictConfig | dict) -> torch.nn.Module:
    """Wrap a model with the configured conformal method."""
    name = get_conformal_name(cfg)
    if name == DEFAULT_CONFORMAL:
        return model
    spec = get_conformal_spec(cfg)
    return spec.transform(model, **conformal_params(cfg))


def build_logit_conformalizer(cfg: DictConfig | dict) -> torch.nn.Module:
    """Build a conformal wrapper around an identity logit model for direct logit fitting."""
    return apply_conformal(torch_identity_logit_model(), cfg)


def fit_logit_conformalizer(
    cfg: DictConfig | dict,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.nn.Module:
    """Fit the configured conformal method directly on logits and labels."""
    logit_conformalizer = build_logit_conformalizer(cfg)
    calibrate(logit_conformalizer, conformal_alpha(cfg), targets, logits)
    return logit_conformalizer


def predict_conformal_sets(conformalizer: torch.nn.Module, logits: torch.Tensor) -> Any:  # noqa: ANN401
    """Apply a fitted logit conformalizer to logits."""
    return predict(conformalizer, logits)


def extract_conformal_metrics(cfg: DictConfig | dict, conformalizer: torch.nn.Module) -> dict[str, float]:
    """Extract conformal-method-specific metrics from a fitted conformalizer."""
    metrics: dict[str, float] = {}
    for extractor in get_conformal_spec(cfg).metric_extractors:
        metrics.update(extractor(conformalizer))
    return metrics


def load_conformal_state(
    model: torch.nn.Module,
    cfg: DictConfig | dict,
    state_dict: Mapping[str, Any],
) -> torch.nn.Module:
    """Load conformal-only state into a conformal model wrapper."""
    spec = get_conformal_spec(cfg)
    missing_state_keys = spec.state_keys.difference(state_dict)
    if missing_state_keys:
        missing = ", ".join(sorted(missing_state_keys))
        msg = f"Conformal artifact is missing required state keys: {missing}."
        raise RuntimeError(msg)

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        unexpected = ", ".join(load_result.unexpected_keys)
        msg = f"Conformal artifact contains unexpected state keys: {unexpected}."
        raise RuntimeError(msg)
    return model
