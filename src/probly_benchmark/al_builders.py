"""Factory functions that construct AL estimators and query strategies from Hydra config.

Bridges the Hydra config world with the probly object world.  The runner
calls :func:`build_al_estimator` and :func:`build_query_strategy` to obtain
the objects needed by the active learning loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from probly.evaluation.active_learning import (
    BADGEQuery,
    MarginSampling,
    RandomQuery,
    TorchEstimator,
    UncertaintyQuery,
    _ProblyEstimator,
)
from probly_benchmark.models import get_base_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

    from probly.evaluation.active_learning.strategies import QueryStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quantifier resolution
# ---------------------------------------------------------------------------

_QUANTIFIER_REGISTRY: dict[str, Callable[[], Callable]] = {}


def _register_quantifier(name: str, factory: Callable[[], Callable]) -> None:
    """Register a lazy quantifier factory under *name*."""
    _QUANTIFIER_REGISTRY[name] = factory


def _lazy_entropy_of_expected_value() -> Callable:
    from probly.quantification.measure.distribution import (  # noqa: PLC0415
        entropy_of_expected_value,
    )

    return entropy_of_expected_value


def _lazy_mutual_information() -> Callable:
    from probly.quantification.measure.distribution import (  # noqa: PLC0415
        mutual_information,
    )

    return mutual_information


def _lazy_entropy() -> Callable:
    from probly.quantification.measure.distribution import entropy  # noqa: PLC0415

    return entropy


def _lazy_upper_entropy() -> Callable:
    from probly.quantification.measure.credal_set import (  # noqa: PLC0415
        upper_entropy,
    )

    return upper_entropy


def _lazy_ddu_entropy() -> Callable:
    from probly.quantification.measure.distribution import entropy  # noqa: PLC0415

    def _ddu_quantifier(rep: Any) -> Any:  # noqa: ANN401
        """Extract softmax from DDU representation, then compute entropy."""
        return entropy(rep.softmax)

    return _ddu_quantifier


_register_quantifier("entropy_of_expected_value", _lazy_entropy_of_expected_value)
_register_quantifier("mutual_information", _lazy_mutual_information)
_register_quantifier("entropy", _lazy_entropy)
_register_quantifier("upper_entropy", _lazy_upper_entropy)
_register_quantifier("ddu_entropy", _lazy_ddu_entropy)


def _resolve_quantifier(name: str | None) -> Callable | None:
    """Look up a quantifier function by config name.

    Args:
        name: Quantifier name as it appears in ``cfg.al_method.quantifier``.

    Returns:
        The resolved callable, or ``None`` when *name* is ``None``.

    Raises:
        ValueError: If the name is not recognised.
    """
    if name is None:
        return None
    key = name.lower()
    if key not in _QUANTIFIER_REGISTRY:
        msg = f"Unknown quantifier: {name!r}"
        raise ValueError(msg)
    return _QUANTIFIER_REGISTRY[key]()


# ---------------------------------------------------------------------------
# Method construction helpers
# ---------------------------------------------------------------------------

_PREDICTOR_TYPE_MAP: dict[str, str] = {
    "ensemble": "logit_classifier",
    "credal_ensembling": "logit_classifier",
    "credal_relative_likelihood": "logit_classifier",
    "dropout": "logit_classifier",
    "ddu": "logit_classifier",
    "efficient_credal_prediction": "logit_classifier",
    "evidential_classification": "logit_classifier",
    "posterior_network": "probabilistic_classifier",
}


def _build_method_predictor(
    representer_name: str,
    base_model_name: str,
    num_classes: int,
    method_params: dict[str, Any],
    *,
    in_features: int | None = None,
) -> Any:  # noqa: ANN401
    """Apply a probly method transformation to a fresh base model.

    Args:
        representer_name: Method name matching a key in
            :data:`probly_benchmark.builders.METHODS`.
        base_model_name: Architecture name forwarded to
            :func:`~probly_benchmark.models.get_base_model`.
        num_classes: Number of output classes.
        method_params: Extra keyword arguments forwarded to the method
            function (e.g. ``n_members``, ``p``).
        in_features: Required for ``tabular_mlp`` models.

    Returns:
        The transformed predictor.
    """
    from probly_benchmark.builders import get_method  # noqa: PLC0415

    method_fn = get_method(representer_name)
    predictor_type = _PREDICTOR_TYPE_MAP.get(representer_name)

    # Posterior network needs a special encoder model and extra args.
    if representer_name == "posterior_network":
        base = get_base_model(
            f"{base_model_name}_encoder",
            num_classes,
            in_features=in_features,
        )
        return method_fn(
            base,
            num_classes=num_classes,
            latent_dim=method_params.pop("latent_dim", 6),
            num_flows=method_params.pop("num_flows", 6),
            class_counts=method_params.pop("class_counts", None),
            predictor_type=predictor_type,
        )

    model_kwargs: dict[str, Any] = {}
    if in_features is not None:
        model_kwargs["in_features"] = in_features

    base = get_base_model(base_model_name, num_classes, **model_kwargs)

    call_kwargs: dict[str, Any] = {}
    if predictor_type is not None:
        call_kwargs["predictor_type"] = predictor_type

    # Forward only the params the method function accepts.
    from probly_benchmark.builders import _filter_params  # noqa: PLC0415

    filtered = _filter_params(method_fn, method_params)
    call_kwargs.update(filtered)

    return method_fn(base, **call_kwargs)


# ---------------------------------------------------------------------------
# Representer construction
# ---------------------------------------------------------------------------


def _build_representer(
    representer_name: str,
    predictor: Any,  # noqa: ANN401
    method_params: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Wrap *predictor* in the appropriate probly representer.

    Args:
        representer_name: The representer/method name from config.
        predictor: A probly-transformed predictor.
        method_params: Extra keyword arguments (e.g. ``alpha``,
            ``distance``) forwarded to the representer constructor.

    Returns:
        A probly representer wrapping the predictor.
    """
    from probly.representer import (  # noqa: PLC0415
        CredalRelativeLikelihoodRepresenter,
        representer,
    )

    if representer_name == "credal_relative_likelihood":
        return CredalRelativeLikelihoodRepresenter(predictor)

    # credal_ensembling needs alpha and distance forwarded.
    if representer_name == "credal_ensembling":
        alpha = method_params.pop("alpha", 0.0)
        distance = method_params.pop("distance", "euclidean")
        return representer(predictor, alpha=alpha, distance=distance)

    return representer(predictor)


# ---------------------------------------------------------------------------
# Method-specific parameter extraction
# ---------------------------------------------------------------------------


def _extract_method_params(cfg: DictConfig) -> dict[str, Any]:
    """Pull method-specific hyperparameters from the AL method config.

    Returns a mutable dict so callers can pop consumed keys.

    Args:
        cfg: The full Hydra config.

    Returns:
        A dictionary of method-specific parameters.
    """
    al = cfg.al_method
    params: dict[str, Any] = {}

    # Ensemble-family methods
    if hasattr(al, "n_members"):
        params["num_members"] = al.n_members

    # Dropout
    if hasattr(al, "p"):
        params["p"] = al.p
    if hasattr(al, "num_samples"):
        params["num_samples"] = al.num_samples

    # DDU
    if hasattr(al, "sn_coeff"):
        params["sn_coeff"] = al.sn_coeff

    # Credal methods
    if hasattr(al, "alpha"):
        params["alpha"] = al.alpha
    if hasattr(al, "distance"):
        params["distance"] = al.distance

    # Evidential
    if hasattr(al, "loss"):
        params["loss"] = al.loss
    if hasattr(al, "kl_weight"):
        params["kl_weight"] = al.kl_weight
    if hasattr(al, "annealing_epochs"):
        params["annealing_epochs"] = al.annealing_epochs

    # Posterior network
    if hasattr(al, "latent_dim"):
        params["latent_dim"] = al.latent_dim
    if hasattr(al, "num_flows"):
        params["num_flows"] = al.num_flows
    if hasattr(al, "entropy_weight"):
        params["entropy_weight"] = al.entropy_weight

    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_uncertainty(_cfg: DictConfig) -> UncertaintyQuery:
    return UncertaintyQuery()


def _build_margin(_cfg: DictConfig) -> MarginSampling:
    return MarginSampling()


def _build_badge(_cfg: DictConfig) -> BADGEQuery:
    return BADGEQuery()


def _build_random(cfg: DictConfig) -> RandomQuery:
    return RandomQuery(seed=cfg.seed)


_STRATEGY_BUILDERS: dict[str, Callable[..., QueryStrategy]] = {
    "uncertainty": _build_uncertainty,
    "margin": _build_margin,
    "badge": _build_badge,
    "random": _build_random,
}


def build_query_strategy(cfg: DictConfig) -> QueryStrategy:
    """Construct a query strategy from Hydra config.

    Args:
        cfg: Hydra config with ``cfg.al_strategy.name``.

    Returns:
        A query strategy instance.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    name = cfg.al_strategy.name.lower()
    builder = _STRATEGY_BUILDERS.get(name)
    if builder is None:
        msg = f"Unknown AL strategy: {cfg.al_strategy.name!r}"
        raise ValueError(msg)
    return builder(cfg)


def build_al_estimator(
    cfg: DictConfig,
    in_features: int | None = None,
) -> TorchEstimator | _ProblyEstimator:
    """Construct an AL estimator from Hydra config.

    For ``plain`` the estimator is a bare :class:`TorchEstimator`.  All other
    methods go through probly's representer/quantifier pipeline and produce a
    :class:`_ProblyEstimator`.

    Args:
        cfg: Full Hydra config.
        in_features: Number of input features, required for ``tabular_mlp``
            base models.

    Returns:
        A ready-to-use estimator with ``fit``, ``predict``,
        ``predict_proba``, and (for non-plain methods)
        ``uncertainty_scores``.

    Raises:
        ValueError: If the method or quantifier name is not recognised.
    """
    method_name: str = cfg.al_method.name
    representer_name: str | None = cfg.al_method.representer
    base_model_name: str = cfg.dataset.base_model
    num_classes: int = cfg.dataset.num_classes

    reset_fn: str | None = "default" if cfg.reset_model else None

    model_kwargs: dict[str, Any] = {}
    if in_features is not None:
        model_kwargs["in_features"] = in_features

    # ---- plain: no UQ, just a standard torch estimator ----
    if method_name == "plain":
        base = get_base_model(base_model_name, num_classes, **model_kwargs)
        return TorchEstimator(
            base,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            device=cfg.device,
            reset_fn=reset_fn,
        )

    # ---- All other methods: probly representer + quantifier ----
    if representer_name is None:
        msg = f"Method {method_name!r} requires a representer but cfg.al_method.representer is None."
        raise ValueError(msg)

    quantifier_fn = _resolve_quantifier(cfg.al_method.quantifier)
    if quantifier_fn is None:
        msg = f"Method {method_name!r} requires a quantifier but cfg.al_method.quantifier is None."
        raise ValueError(msg)

    method_params = _extract_method_params(cfg)

    # Build the probly predictor and representer.
    predictor = _build_method_predictor(
        representer_name,
        base_model_name,
        num_classes,
        method_params,
        in_features=in_features,
    )
    rep = _build_representer(representer_name, predictor, method_params)

    logger.info(
        "Built _ProblyEstimator: method=%s representer=%s quantifier=%s",
        method_name,
        representer_name,
        cfg.al_method.quantifier,
    )

    return _ProblyEstimator(
        rep,
        quantifier_fn,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        reset_fn=reset_fn,
    )
