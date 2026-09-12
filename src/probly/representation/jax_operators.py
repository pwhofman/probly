"""Python operators expressed through probly's JAX function override protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from probly.representation import jax_functions as jf

if TYPE_CHECKING:
    from collections.abc import Callable


def _binary(func: Callable, *, reflected: bool = False) -> Callable[[Any, object], Any]:
    """Make an operator that leaves unsupported operands to Python's fallback."""

    def method(self: object, other: object) -> Any:  # noqa: ANN401
        """Dispatch the binary operation or return NotImplemented.

        Args:
            self: The operand providing this operator.
            other: The other operand.

        Returns:
            The override result, or NotImplemented for unsupported operands.
        """
        args = (other, self) if reflected else (self, other)
        return jf.try_jax_function(func, args, *args)

    return method


def _unary(func: Callable) -> Callable[[Any], Any]:
    """Make a unary operator, for which Python has no reflected fallback."""

    def method(self: object) -> Any:  # noqa: ANN401
        """Return the unary operation's override result."""
        return jf.handle_jax_function(func, (self,), self)

    return method


class JaxOperatorsMixin:  # noqa: PLW1641
    """Provide array operators through ``__jax_function__`` implementations.

    Operators use the corresponding ``jax_*`` wrapper as their dispatch key.
    Implementations control permissions and result types. Augmented assignment
    uses Python's ordinary out-of-place operator fallback, matching JAX immutability.
    Subclasses can override comparison methods; dataclasses must use ``eq=False``
    to inherit the mixin's equality operator instead of generating their own.
    """

    # NumPy must defer mixed infix expressions rather than coerce a protected
    # operand through __array__ and discard its wrapper and permissions.
    __array_priority__ = 1000
    __array_ufunc__ = None

    __add__ = _binary(jf.jax_add)
    __radd__ = _binary(jf.jax_add, reflected=True)
    __sub__ = _binary(jf.jax_subtract)
    __rsub__ = _binary(jf.jax_subtract, reflected=True)
    __mul__ = _binary(jf.jax_multiply)
    __rmul__ = _binary(jf.jax_multiply, reflected=True)
    __truediv__ = _binary(jf.jax_true_divide)
    __rtruediv__ = _binary(jf.jax_true_divide, reflected=True)
    __floordiv__ = _binary(jf.jax_floor_divide)
    __rfloordiv__ = _binary(jf.jax_floor_divide, reflected=True)
    __mod__ = _binary(jf.jax_remainder)
    __rmod__ = _binary(jf.jax_remainder, reflected=True)
    __divmod__ = _binary(jf.jax_divmod)
    __rdivmod__ = _binary(jf.jax_divmod, reflected=True)
    __matmul__ = _binary(jf.jax_matmul)
    __rmatmul__ = _binary(jf.jax_matmul, reflected=True)
    __and__ = _binary(jf.jax_bitwise_and)
    __rand__ = _binary(jf.jax_bitwise_and, reflected=True)
    __or__ = _binary(jf.jax_bitwise_or)
    __ror__ = _binary(jf.jax_bitwise_or, reflected=True)
    __xor__ = _binary(jf.jax_bitwise_xor)
    __rxor__ = _binary(jf.jax_bitwise_xor, reflected=True)
    __lshift__ = _binary(jf.jax_left_shift)
    __rlshift__ = _binary(jf.jax_left_shift, reflected=True)
    __rshift__ = _binary(jf.jax_right_shift)
    __rrshift__ = _binary(jf.jax_right_shift, reflected=True)
    __eq__ = _binary(jf.jax_equal)
    __ne__ = _binary(jf.jax_not_equal)
    __lt__ = _binary(jf.jax_less)
    __le__ = _binary(jf.jax_less_equal)
    __gt__ = _binary(jf.jax_greater)
    __ge__ = _binary(jf.jax_greater_equal)
    __pos__ = _unary(jf.jax_positive)
    __neg__ = _unary(jf.jax_negative)
    __abs__ = _unary(jf.jax_absolute)
    __invert__ = _unary(jf.jax_invert)

    def __pow__(self, other: object, modulo: object = None) -> Any:  # noqa: ANN401
        """Dispatch exponentiation; modular exponentiation is unsupported."""
        if modulo is not None:
            return NotImplemented
        return jf.try_jax_function(jf.jax_power, (self, other), self, other)

    def __rpow__(self, other: object, modulo: object = None) -> Any:  # noqa: ANN401
        """Dispatch reflected exponentiation; modular exponentiation is unsupported."""
        if modulo is not None:
            return NotImplemented
        return jf.try_jax_function(jf.jax_power, (other, self), other, self)
