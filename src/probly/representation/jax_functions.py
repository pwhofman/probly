# pyright: reportInvalidTypeForm=false
"""Override machinery and ``jax.numpy`` mirroring wrappers.

``jnp.reshape(custom_obj, ...)`` cannot be intercepted by a custom array-like object: JAX has no
override protocol, ``__jax_array__`` is no longer honored, and pytree registration only covers
``jax.jit``, ``jax.vmap`` and ``jax.tree``. This module supplies the missing piece: a
``__jax_function__`` hook with :func:`has_jax_function` and :func:`handle_jax_function`, plus thin
wrappers over the ``jax.numpy`` functions probly needs. Call ``jax_reshape(x, ...)`` instead of
``jnp.reshape(x, ...)`` so custom objects get a say.

Dispatch duck-types on ``__jax_function__`` being defined on the argument's *type*, so this module
never imports :mod:`probly.representation.jax_like` and the import graph stays one-way.

The :func:`jax_function_dispatch` decorator selects operands by parameter name and preserves
public signatures and documentation. Operator overloads give native operands precise array
result types; arbitrary :class:`SupportsJaxFunction` overrides have an ``object`` result type.
Their native implementation bodies run only after dispatch has excluded custom overrides,
which is why narrowing casts to native operand types are safe there.
"""

from __future__ import annotations

from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import jax
    from jax.sharding import Sharding
    from jax.typing import ArrayLike as JaxArrayLike, DTypeLike


class SupportsJaxFunction(Protocol):
    """Structural interface for custom JAX function overrides.

    An override may return a representation, an array, or another result type.
    This interface deliberately does not promise that operations preserve type.
    """

    @classmethod
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        """Return an override result or NotImplemented for an unsupported call."""
        ...


type JaxOperand = JaxArrayLike | SupportsJaxFunction


class _SignaturePreservingDecorator(Protocol):
    def __call__[F: Callable[..., object]](self, implementation: F, /) -> F: ...


def jax_function_dispatch(*names: str, unpack: tuple[str, ...] = ()) -> _SignaturePreservingDecorator:
    """Decorate a native implementation with JAX function override dispatch.

    Args:
        *names: Names of individual parameters participating in dispatch.
        unpack: Names of sequence parameters whose elements participate in
            dispatch. These parameters must contain reusable iterables.

    Returns:
        A decorator preserving the implementation's signature, annotations,
        and documentation. Overrides receive the decorated public function as
        their dispatch key and the original positional and keyword arguments.

    Raises:
        ValueError: If selectors are empty, repeated, unknown, or variadic.

    Examples:
        Use ``@jax_function_dispatch("a", "where")`` for a reduction, or
        ``@jax_function_dispatch(unpack=("arrays",))`` for a sequence operation.
    """
    selectors = (*names, *unpack)
    if not selectors or len(set(selectors)) != len(selectors):
        msg = "Dispatch selectors must be nonempty and unique."
        raise ValueError(msg)

    def decorator[F: Callable[..., object]](implementation: F) -> F:
        function_signature = signature(implementation)
        for name in selectors:
            parameter = function_signature.parameters.get(name)
            if parameter is None or parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                msg = f"Dispatch selector {name!r} must name a non-variadic parameter."
                raise ValueError(msg)

        @wraps(implementation)
        def wrapper(*args: object, **kwargs: object) -> object:
            bound = function_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            relevant_args = [bound.arguments[name] for name in names]
            for name in unpack:
                relevant_args.extend(bound.arguments[name])
            if has_jax_function(relevant_args):
                return handle_jax_function(wrapper, relevant_args, *args, **kwargs)
            return implementation(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def _overloaded_args(relevant_args: Iterable[object]) -> list[object]:
    """Collect the arguments whose type defines ``__jax_function__``, subclasses first."""
    overloaded: list[object] = []
    overloaded_types: list[type[Any]] = []

    for arg in relevant_args:
        arg_type = type(arg)
        if arg is None or not hasattr(arg_type, "__jax_function__") or arg_type in overloaded_types:
            continue

        # Subclasses get a chance to handle the call before their base classes.
        index = len(overloaded_types)
        for position, other in enumerate(overloaded_types):
            if issubclass(arg_type, other):
                index = position
                break

        overloaded_types.insert(index, arg_type)
        overloaded.insert(index, arg)

    return overloaded


def has_jax_function(relevant_args: Iterable[object]) -> bool:
    """Check whether any of the arguments overrides the jax function protocol.

    Args:
        relevant_args: The arguments to inspect. ``None`` entries are ignored.

    Returns:
        True if the type of at least one argument defines ``__jax_function__``.
    """
    return any(arg is not None and hasattr(type(arg), "__jax_function__") for arg in relevant_args)


def handle_jax_function(
    public_api: Callable[..., Any],
    relevant_args: Iterable[object],
    *args: object,
    **kwargs: object,
) -> Any:  # noqa: ANN401
    """Dispatch a call to the ``__jax_function__`` implementation of the arguments.

    Args:
        public_api: The wrapper function that was called, used as the dispatch key.
        relevant_args: The arguments that may override the call.
        *args: The positional arguments the wrapper was called with.
        **kwargs: The keyword arguments the wrapper was called with.

    Returns:
        The first result that is not ``NotImplemented``.

    Raises:
        TypeError: If no argument implements ``public_api``.
    """
    relevant_args = tuple(relevant_args)
    result = try_jax_function(public_api, relevant_args, *args, **kwargs)
    if result is not NotImplemented:
        return result

    types = [type(arg) for arg in _overloaded_args(relevant_args)]
    name = getattr(public_api, "__name__", repr(public_api))
    msg = f"no implementation found for {name} on types that implement __jax_function__: {types}"
    raise TypeError(msg)


def try_jax_function(
    public_api: Callable[..., Any],
    relevant_args: Iterable[object],
    *args: object,
    **kwargs: object,
) -> Any:  # noqa: ANN401
    """Try function overrides without preventing Python's reflected operator fallback.

    Args:
        public_api: Wrapper function used as the dispatch key.
        relevant_args: Arguments whose types may override the function.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The first implemented result, or NotImplemented if every override declines.
        Exceptions raised by implementations propagate unchanged.
    """
    overloaded = _overloaded_args(relevant_args)
    types = tuple(type(arg) for arg in overloaded)

    for arg in overloaded:
        result = type(arg).__jax_function__(public_api, types, args, kwargs)  # ty: ignore[unresolved-attribute]
        if result is not NotImplemented:
            return result

    return NotImplemented


@overload
def jax_add(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_add(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_add(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Add two operands elementwise, mirroring ``jax.numpy.add``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        The elementwise sum, or the result supplied by a custom override.
    """
    return jnp.add(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_subtract(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_subtract(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_subtract(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Subtract operands elementwise, mirroring ``jax.numpy.subtract``.

    Args:
        x1: Operand from which to subtract.
        x2: Operand to subtract.

    Returns:
        ``x1 - x2`` elementwise, or the result supplied by a custom override.
    """
    return jnp.subtract(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_multiply(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_multiply(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_multiply(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Multiply operands elementwise, mirroring ``jax.numpy.multiply``.

    Args:
        x1: Left factor.
        x2: Right factor.

    Returns:
        The elementwise product, or the result supplied by a custom override.
    """
    return jnp.multiply(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_true_divide(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_true_divide(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_true_divide(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Divide operands elementwise, mirroring ``jax.numpy.true_divide``.

    Args:
        x1: Dividend.
        x2: Divisor.

    Returns:
        The true quotient ``x1 / x2``, or a custom override result.
    """
    return jnp.true_divide(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_floor_divide(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_floor_divide(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_floor_divide(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Floor-divide operands, mirroring ``jax.numpy.floor_divide``.

    Args:
        x1: Dividend.
        x2: Divisor.

    Returns:
        The quotient rounded toward negative infinity, or a custom override result.
    """
    return jnp.floor_divide(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_remainder(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_remainder(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_remainder(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compute floor-division remainders, mirroring ``jax.numpy.remainder``.

    Args:
        x1: Dividend.
        x2: Divisor, whose sign determines the remainder's sign.

    Returns:
        The elementwise remainder, or the result supplied by a custom override.
    """
    return jnp.remainder(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_divmod(x1: JaxArrayLike, x2: JaxArrayLike, /) -> tuple[jax.Array, jax.Array]: ...


@overload
def jax_divmod(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_divmod(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compute quotients and remainders, mirroring ``jax.numpy.divmod``.

    Args:
        x1: Dividend.
        x2: Divisor.

    Returns:
        A tuple of floor-division quotients and remainders for native operands,
        or the result supplied by a custom override.
    """
    return jnp.divmod(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_power(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_power(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_power(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Raise operands to powers elementwise, mirroring ``jax.numpy.power``.

    Args:
        x1: Bases.
        x2: Exponents.

    Returns:
        ``x1 ** x2`` elementwise, or the result supplied by a custom override.
    """
    return jnp.power(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_matmul(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_matmul(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_matmul(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Multiply matrices or vectors, mirroring ``jax.numpy.matmul``.

    Args:
        x1: Left matrix or vector operand.
        x2: Right matrix or vector operand.

    Returns:
        The matrix product with broadcast batch dimensions for native operands,
        or the result supplied by a custom override.
    """
    return jnp.matmul(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_bitwise_and(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_bitwise_and(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_bitwise_and(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compute bitwise AND, mirroring ``jax.numpy.bitwise_and``.

    Args:
        x1: Left integer or boolean operand.
        x2: Right integer or boolean operand.

    Returns:
        The elementwise bitwise AND, or a custom override result.
    """
    return jnp.bitwise_and(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_bitwise_or(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_bitwise_or(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_bitwise_or(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compute bitwise OR, mirroring ``jax.numpy.bitwise_or``.

    Args:
        x1: Left integer or boolean operand.
        x2: Right integer or boolean operand.

    Returns:
        The elementwise bitwise OR, or a custom override result.
    """
    return jnp.bitwise_or(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_bitwise_xor(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_bitwise_xor(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_bitwise_xor(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compute bitwise exclusive OR, mirroring ``jax.numpy.bitwise_xor``.

    Args:
        x1: Left integer or boolean operand.
        x2: Right integer or boolean operand.

    Returns:
        The elementwise bitwise exclusive OR, or a custom override result.
    """
    return jnp.bitwise_xor(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_left_shift(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_left_shift(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_left_shift(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Shift bits left elementwise, mirroring ``jax.numpy.left_shift``.

    Args:
        x1: Integer values to shift.
        x2: Number of bit positions to shift each value.

    Returns:
        The left-shifted values, or the result supplied by a custom override.
    """
    return jnp.left_shift(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_right_shift(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_right_shift(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_right_shift(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Shift bits right elementwise, mirroring ``jax.numpy.right_shift``.

    Args:
        x1: Integer values to shift.
        x2: Number of bit positions to shift each value.

    Returns:
        The right-shifted values, or the result supplied by a custom override.
    """
    return jnp.right_shift(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_equal(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_equal(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_equal(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compare operands for equality, mirroring ``jax.numpy.equal``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        An elementwise boolean equality array for native operands, or a custom
        override result. Protected-axis overrides return batch-shaped masks.
    """
    return jnp.equal(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_not_equal(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_not_equal(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_not_equal(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Compare operands for inequality, mirroring ``jax.numpy.not_equal``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        An elementwise boolean inequality array for native operands, or a custom
        override result. Protected-axis overrides return batch-shaped masks.
    """
    return jnp.not_equal(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_less(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_less(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_less(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Test whether the left operand is smaller, mirroring ``jax.numpy.less``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        The elementwise boolean result of ``x1 < x2`` for native operands,
        or the result supplied by a custom override.
    """
    return jnp.less(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_less_equal(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_less_equal(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_less_equal(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Test the less-than-or-equal relation, mirroring ``jax.numpy.less_equal``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        The elementwise boolean result of ``x1 <= x2`` for native operands,
        or the result supplied by a custom override.
    """
    return jnp.less_equal(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_greater(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_greater(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_greater(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Test whether the left operand is larger, mirroring ``jax.numpy.greater``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        The elementwise boolean result of ``x1 > x2`` for native operands,
        or the result supplied by a custom override.
    """
    return jnp.greater(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_greater_equal(x1: JaxArrayLike, x2: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_greater_equal(x1: JaxOperand, x2: JaxOperand, /) -> object: ...


@jax_function_dispatch("x1", "x2")
def jax_greater_equal(x1: JaxOperand, x2: JaxOperand, /) -> object:
    """Test the greater-than-or-equal relation, mirroring ``jax.numpy.greater_equal``.

    Args:
        x1: Left operand.
        x2: Right operand.

    Returns:
        The elementwise boolean result of ``x1 >= x2`` for native operands,
        or the result supplied by a custom override.
    """
    return jnp.greater_equal(cast("JaxArrayLike", x1), cast("JaxArrayLike", x2))


@overload
def jax_positive(x: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_positive(x: JaxOperand, /) -> object: ...


@jax_function_dispatch("x")
def jax_positive(x: JaxOperand, /) -> object:
    """Apply unary positive, mirroring ``jax.numpy.positive``.

    Args:
        x: Operand.

    Returns:
        The positive array, or the result supplied by a custom override.
    """
    return jnp.positive(cast("JaxArrayLike", x))


@overload
def jax_negative(x: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_negative(x: JaxOperand, /) -> object: ...


@jax_function_dispatch("x")
def jax_negative(x: JaxOperand, /) -> object:
    """Negate an operand elementwise, mirroring ``jax.numpy.negative``.

    Args:
        x: Operand to negate.

    Returns:
        The elementwise negation, or the result supplied by a custom override.
    """
    return jnp.negative(cast("JaxArrayLike", x))


@overload
def jax_absolute(x: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_absolute(x: JaxOperand, /) -> object: ...


@jax_function_dispatch("x")
def jax_absolute(x: JaxOperand, /) -> object:
    """Compute absolute values elementwise, mirroring ``jax.numpy.absolute``.

    Args:
        x: Real or complex operand.

    Returns:
        The elementwise magnitudes, or the result supplied by a custom override.
    """
    return jnp.absolute(cast("JaxArrayLike", x))


@overload
def jax_invert(x: JaxArrayLike, /) -> jax.Array: ...


@overload
def jax_invert(x: JaxOperand, /) -> object: ...


@jax_function_dispatch("x")
def jax_invert(x: JaxOperand, /) -> object:
    """Invert bits elementwise, mirroring ``jax.numpy.invert``.

    Args:
        x: Integer or boolean operand.

    Returns:
        The bitwise complement, or the result supplied by a custom override.
    """
    return jnp.invert(cast("JaxArrayLike", x))


@jax_function_dispatch("a")
def jax_reshape(
    a: JaxArrayLike,
    /,
    shape: int | Sequence[int],
    order: str = "C",
    *,
    copy: bool | None = None,
) -> jax.Array:
    """Return a reshaped version of the array, mirroring ``jax.numpy.reshape``.

    Args:
        a: The array to reshape.
        shape: The target shape.
        order: The read/write element order.
        copy: Whether to copy the data.

    Returns:
        The reshaped array.
    """
    return jnp.reshape(a, shape, order, copy=copy)


@jax_function_dispatch("a")
def jax_transpose(a: JaxArrayLike, /, axes: Sequence[int] | None = None) -> jax.Array:
    """Return a transposed version of the array, mirroring ``jax.numpy.transpose``.

    Args:
        a: The array to transpose.
        axes: The permutation of axes. If omitted, the axes are reversed.

    Returns:
        The transposed array.
    """
    return jnp.transpose(a, axes)


@jax_function_dispatch("x")
def jax_matrix_transpose(x: JaxArrayLike, /) -> jax.Array:
    """Transpose the last two axes of the array, mirroring ``jax.numpy.matrix_transpose``.

    Args:
        x: The array to transpose.

    Returns:
        The array with its last two axes swapped.
    """
    return jnp.matrix_transpose(x)


@jax_function_dispatch("a")
def jax_moveaxis(
    a: JaxArrayLike,
    /,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> jax.Array:
    """Move axes of the array to new positions, mirroring ``jax.numpy.moveaxis``.

    Args:
        a: The array to move axes of.
        source: The axes to move.
        destination: The destination positions of the moved axes.

    Returns:
        The array with the axes moved.
    """
    return jnp.moveaxis(a, source, destination)


@jax_function_dispatch("a")
def jax_swapaxes(a: JaxArrayLike, /, axis1: int, axis2: int) -> jax.Array:
    """Swap two axes of the array, mirroring ``jax.numpy.swapaxes``.

    Args:
        a: The array to swap axes of.
        axis1: The first axis.
        axis2: The second axis.

    Returns:
        The array with the two axes swapped.
    """
    return jnp.swapaxes(a, axis1, axis2)


@jax_function_dispatch("a")
def jax_squeeze(a: JaxArrayLike, /, axis: int | Sequence[int] | None = None) -> jax.Array:
    """Remove axes of length one, mirroring ``jax.numpy.squeeze``.

    Args:
        a: The array to squeeze.
        axis: The axis or axes to remove. If omitted, all length-one axes are removed.

    Returns:
        The squeezed array.
    """
    return jnp.squeeze(a, axis)


@jax_function_dispatch("a")
def jax_expand_dims(a: JaxArrayLike, /, axis: int | Sequence[int]) -> jax.Array:
    """Insert axes of length one, mirroring ``jax.numpy.expand_dims``.

    Args:
        a: The array to expand.
        axis: The position or positions of the new axes.

    Returns:
        The expanded array.
    """
    return jnp.expand_dims(a, axis)


@jax_function_dispatch(unpack=("arrays",))
def jax_concatenate(
    arrays: Sequence[JaxArrayLike],
    /,
    axis: int | None = 0,
    dtype: DTypeLike | None = None,
) -> jax.Array:
    """Join a sequence of arrays along an existing axis, mirroring ``jax.numpy.concatenate``.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis to concatenate along. ``None`` flattens the inputs first.
        dtype: The desired data type of the result.

    Returns:
        The concatenated array.
    """
    return jnp.concatenate(arrays, axis, dtype=dtype)


@jax_function_dispatch(unpack=("arrays",))
def jax_stack(
    arrays: Sequence[JaxArrayLike],
    /,
    axis: int = 0,
    out: None = None,
    dtype: DTypeLike | None = None,
) -> jax.Array:
    """Join a sequence of arrays along a new axis, mirroring ``jax.numpy.stack``.

    Args:
        arrays: The arrays to stack.
        axis: The axis of the result along which the inputs are stacked.
        out: Unsupported by JAX, must be ``None``.
        dtype: The desired data type of the result.

    Returns:
        The stacked array.
    """
    return jnp.stack(arrays, axis, out, dtype=dtype)


@jax_function_dispatch("arr", "indices")
def jax_take_along_axis(
    arr: JaxArrayLike,
    /,
    indices: JaxArrayLike,
    axis: int | None = -1,
    mode: str | None = None,
    fill_value: Any = None,  # noqa: ANN401
) -> jax.Array:
    """Take values along an axis, mirroring ``jax.numpy.take_along_axis``.

    Args:
        arr: The array to take values from.
        indices: The indices to take.
        axis: The axis to take along. ``None`` flattens the input first.
        mode: How out-of-bounds indices are handled.
        fill_value: The value used for out-of-bounds indices in ``"fill"`` mode.

    Returns:
        The gathered array.
    """
    return jnp.take_along_axis(arr, indices, axis, mode=mode, fill_value=fill_value)


@jax_function_dispatch("x")
def jax_conj(x: JaxArrayLike, /) -> jax.Array:
    """Return the complex conjugate of the array, mirroring ``jax.numpy.conj``.

    Args:
        x: The array to conjugate.

    Returns:
        The conjugated array.
    """
    return jnp.conj(x)


@jax_function_dispatch("a")
def jax_copy(a: JaxArrayLike, /, order: str | None = None) -> jax.Array:
    """Return a copy of the array, mirroring ``jax.numpy.copy``.

    Args:
        a: The array to copy.
        order: The memory layout of the copy.

    Returns:
        The copied array.
    """
    return jnp.copy(a, order)


@jax_function_dispatch("x")
def jax_astype(
    x: JaxArrayLike,
    /,
    dtype: DTypeLike | None,
    *,
    copy: bool = False,
    device: jax.Device | Sharding | None = None,
) -> jax.Array:
    """Cast the array to a new data type, mirroring ``jax.numpy.astype``.

    Args:
        x: The array to cast.
        dtype: The target data type.
        copy: Whether to always return a copy.
        device: The device the result should live on.

    Returns:
        The cast array.
    """
    return jnp.astype(x, dtype, copy=copy, device=device)


@jax_function_dispatch("a", "where")
def jax_mean(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the arithmetic mean, mirroring ``jax.numpy.mean``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The mean of the array.
    """
    return jnp.mean(a, axis, dtype, out, keepdims, where=where)


@jax_function_dispatch("a", "initial", "where")
def jax_sum(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: JaxArrayLike | None = None,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the sum, mirroring ``jax.numpy.sum``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        keepdims: Whether reduced axes are retained with size one.
        initial: The starting value of the reduction.
        where: Optional mask selecting the elements to include.

    Returns:
        The sum of the array.
    """
    return jnp.sum(a, axis, dtype, out, keepdims, initial, where)


@jax_function_dispatch("a", "where")
def jax_std(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the standard deviation, mirroring ``jax.numpy.std``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        ddof: The delta degrees of freedom.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The standard deviation of the array.
    """
    return jnp.std(a, axis, dtype, out, ddof, keepdims, where=where)


@jax_function_dispatch("a", "where")
def jax_var(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: JaxArrayLike | None = None,
) -> jax.Array:
    """Compute the variance, mirroring ``jax.numpy.var``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        dtype: The data type of the accumulator.
        out: Unsupported by JAX, must be ``None``.
        ddof: The delta degrees of freedom.
        keepdims: Whether reduced axes are retained with size one.
        where: Optional mask selecting the elements to include.

    Returns:
        The variance of the array.
    """
    return jnp.var(a, axis, dtype, out, ddof, keepdims, where=where)


@jax_function_dispatch("a", "weights")
def jax_average(
    a: JaxArrayLike,
    /,
    axis: int | Sequence[int] | None = None,
    weights: JaxArrayLike | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute a possibly weighted average, mirroring ``jax.numpy.average``.

    Args:
        a: The array to reduce.
        axis: The axis or axes to reduce.
        weights: Optional weights, broadcastable to ``a`` or matching the reduced axis.
        returned: Whether to also return the sum of the weights.
        keepdims: Whether reduced axes are retained with size one.

    Returns:
        The weighted average, and the sum of the weights if ``returned`` is True.
    """
    return jnp.average(a, axis, weights, returned, keepdims)


__all__ = [
    "JaxOperand",
    "SupportsJaxFunction",
    "handle_jax_function",
    "has_jax_function",
    "jax_absolute",
    "jax_add",
    "jax_astype",
    "jax_average",
    "jax_bitwise_and",
    "jax_bitwise_or",
    "jax_bitwise_xor",
    "jax_concatenate",
    "jax_conj",
    "jax_copy",
    "jax_divmod",
    "jax_equal",
    "jax_expand_dims",
    "jax_floor_divide",
    "jax_function_dispatch",
    "jax_greater",
    "jax_greater_equal",
    "jax_invert",
    "jax_left_shift",
    "jax_less",
    "jax_less_equal",
    "jax_matmul",
    "jax_matrix_transpose",
    "jax_mean",
    "jax_moveaxis",
    "jax_multiply",
    "jax_negative",
    "jax_not_equal",
    "jax_positive",
    "jax_power",
    "jax_remainder",
    "jax_reshape",
    "jax_right_shift",
    "jax_squeeze",
    "jax_stack",
    "jax_std",
    "jax_subtract",
    "jax_sum",
    "jax_swapaxes",
    "jax_take_along_axis",
    "jax_transpose",
    "jax_true_divide",
    "jax_var",
    "try_jax_function",
]
