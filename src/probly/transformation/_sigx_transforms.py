"""Stub transform callbacks for predictor transformation decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sigx_gen.builder import SignatureBuilder

from probly.predictor import predictor_registry

if TYPE_CHECKING:
    from sigx_gen.model.transform_api import TransformFactoryContext, TransformResult


_BROAD_PREDICTOR_TYPE_ANNOTATION = "probly.predictor.PredictorName | type[probly.predictor.Predictor] | None"


def _build_registry_names_by_type() -> dict[type[object], tuple[str, ...]]:
    """Build predictor-type to registered-name mapping from predictor_registry."""
    names_by_type: dict[type[object], set[str]] = {}
    for name, predictor_type in predictor_registry.items():
        if isinstance(predictor_type, type):
            names_by_type.setdefault(predictor_type, set()).add(name)
    return {predictor_type: tuple(sorted(names)) for predictor_type, names in names_by_type.items()}


_REGISTRY_NAMES_BY_TYPE = _build_registry_names_by_type()


def _explicit_predictor_types(value: object) -> tuple[type[object], ...]:
    """Collect explicit predictor classes from decorator factory arguments."""
    if isinstance(value, type):
        return (value,)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(item for item in value if isinstance(item, type))
    return ()


def _predictor_type_ref(predictor_type: type[object]) -> str:
    """Render fully-qualified type reference for annotation output."""
    if predictor_type.__module__.startswith("probly.predictor"):
        return f"probly.predictor.{predictor_type.__qualname__}"
    return f"{predictor_type.__module__}.{predictor_type.__qualname__}"


def _predictor_type_annotation_from_context(ctx: TransformFactoryContext) -> str:
    """Build a precise predictor_type annotation from factory arguments."""
    permitted = ctx.bound_factory_args.arguments.get("permitted_predictor_types")
    if permitted is None:
        return _BROAD_PREDICTOR_TYPE_ANNOTATION

    explicit_types = tuple(sorted(set(_explicit_predictor_types(permitted)), key=_predictor_type_ref))
    if not explicit_types:
        return _BROAD_PREDICTOR_TYPE_ANNOTATION

    literal_names = sorted({name for typ in explicit_types for name in _REGISTRY_NAMES_BY_TYPE.get(typ, ())})

    annotation_parts: list[str] = []
    if literal_names:
        literal_values = ", ".join(repr(name) for name in literal_names)
        annotation_parts.append(f"Literal[{literal_values}]")

    annotation_parts.extend(f"type[{_predictor_type_ref(typ)}]" for typ in explicit_types)
    annotation_parts.append("None")
    return " | ".join(annotation_parts)


def predictor_transformation_transform(ctx: TransformFactoryContext) -> TransformResult:
    """Augment transformed signatures with the optional predictor type selector."""
    builder = SignatureBuilder.from_signature(ctx.original)
    builder.add_kwonly(
        "predictor_type",
        annotation=_predictor_type_annotation_from_context(ctx),
        default="None",
        if_missing=True,
    )
    return builder.build()
