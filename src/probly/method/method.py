"""Utilities for the definition of methods."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from probly.predictor import Predictor, PredictorName, predictor_registry

F = TypeVar("F")

try:
    from sigx import stub_transform_factory
except ImportError:

    def stub_transform_factory(_transform_ref: str) -> Callable[[F], F]:
        """Return a no-op transform marker decorator when sigx is unavailable."""

        def decorator(func: F) -> F:
            return func

        return decorator


if TYPE_CHECKING:
    from collections.abc import Callable, Collection


class PredictorTransformationMethod[PIn: Predictor, **In, POut: Predictor](Protocol):
    """Protocol for methods."""

    def __call__(self, base: PIn, *args: In.args, **kwargs: In.kwargs) -> POut:
        """Call the method."""


@stub_transform_factory("probly.method._sigx_transforms:predictor_transformation_transform")
def predictor_transformation[Pin: Predictor, **In, POut: Predictor](
    permitted_predictor_types: Collection[type[Predictor]] | None,
    preserve_predictor_type: bool = True,
    auto_infer_predictor_type: bool = True,
) -> Callable[[PredictorTransformationMethod[Pin, In, POut]], PredictorTransformationMethod[Pin, In, POut]]:
    """Decorator factory for predictor transformation methods.

    Args:
        permitted_predictor_types: Optional collection of predictor types that the method can be applied to.
            If None, the method can be applied to any predictor type.
        preserve_predictor_type: Whether to preserve the originalpredictor type of the transformed predictor.
            Default is True.
        auto_infer_predictor_type: Whether to automatically infer the predictor type if not explicitly specified.
            Default is True.

    Returns:
        A decorator that transforms a predictor transformation method into a method
        that can be applied to predictors of the specified types.
        Untyped predictors can be typed by specifying the predictor type via the `predictor_type`
        keyword argument when calling the transformation method.
    """

    def decorator[Pin: Predictor, **In, POut: Predictor](
        func: PredictorTransformationMethod[Pin, In, POut],
    ) -> PredictorTransformationMethod[Pin, In, POut]:

        @functools.wraps(func)
        def wrapper(
            base: Pin,
            *args: Any,  # noqa: ANN401
            predictor_type: PredictorName | type[Predictor] | None = None,
            **kwargs: Any,  # noqa: ANN401
        ) -> POut:
            inferred_type = None
            if permitted_predictor_types is not None:
                if len(permitted_predictor_types) == 1 and auto_infer_predictor_type:
                    inferred_type = next(iter(permitted_predictor_types))
                else:
                    for t in permitted_predictor_types:
                        if isinstance(base, t):
                            inferred_type = t
                            break

            if predictor_type is not None:
                predictor_type = (
                    predictor_registry[predictor_type] if isinstance(predictor_type, str) else predictor_type
                )

                if inferred_type is not None and predictor_type is not inferred_type:
                    msg = f"Explicit predictor type {predictor_type} does not match inferred type {inferred_type}."
                    raise ValueError(msg)
            else:
                predictor_type = inferred_type

            if predictor_type is None:
                msg = "Could not determine predictor type. Please specify predictor_type. Supported types: "
                for predictor in permitted_predictor_types or predictor_registry.values():
                    msg += f"{predictor.__name__}, "
                raise ValueError(msg)

            base = predictor_type.register_instance(base)
            res = func(base, *args, **kwargs)

            if preserve_predictor_type:
                res = predictor_type.register_instance(res)

            return res

        return wrapper

    return decorator
