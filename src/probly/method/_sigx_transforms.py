"""Stub transform callbacks for predictor transformation decorators."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from sigx_gen.builder import SignatureBuilder

if TYPE_CHECKING:
    from sigx_gen.model.transform_api import TransformFactoryContext, TransformResult


_BROAD_PREDICTOR_TYPE_ANNOTATION = "probly.predictor.PredictorName | type[probly.predictor.Predictor] | None"


@lru_cache(maxsize=1)
def _predictor_registry_name_map() -> dict[str, tuple[str, ...]]:
    """Build class-name to registered predictor-name map from predictor definitions."""
    predictor_file = Path(__file__).resolve().parents[1] / "predictor" / "_common.py"
    if not predictor_file.is_file():
        return {}

    module = ast.parse(predictor_file.read_text(encoding="utf-8"), filename=str(predictor_file))
    name_map: dict[str, tuple[str, ...]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue

        registered_names: set[str] = set()
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "multi_register"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "predictor_registry"
            ):
                continue
            if not decorator.args:
                continue

            names = decorator.args[0]
            if not isinstance(names, (ast.List, ast.Tuple)):
                continue
            for element in names.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    registered_names.add(element.value)

        if registered_names:
            name_map[node.name] = tuple(sorted(registered_names))

    return name_map


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

    registry_map = _predictor_registry_name_map()
    literal_names = sorted({name for typ in explicit_types for name in registry_map.get(typ.__name__, ())})

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
