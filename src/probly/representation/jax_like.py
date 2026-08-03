"""Protocols and base classes for JAX-like array objects.

This module mirrors :mod:`probly.representation.torch_like`. The one structural difference is
that JAX has no counterpart to ``torch.overrides``: ``jax.numpy`` functions cannot be intercepted
by custom objects. The override machinery therefore lives in
:mod:`probly.representation.jax_functions`, and the concrete methods below are written in terms of
its wrappers so that subclasses only have to implement ``__jax_function__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import fields, is_dataclass
from inspect import isabstract
from typing import TYPE_CHECKING, Any, Protocol, Self, override, runtime_checkable

from flextype import flexdispatch
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np

from probly.representation.array_like import ArrayLike
from probly.representation.jax_functions import (
    handle_jax_function,
    has_jax_function,
    jax_astype,
    jax_conj,
    jax_copy,
    jax_matrix_transpose,
    jax_reshape,
    jax_squeeze,
    jax_transpose,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax.sharding import Sharding
    from jax.typing import DTypeLike
    from numpy.typing import DTypeLike as NpDTypeLike, NDArray


@runtime_checkable
class JaxLikeConvertible[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that can be converted to JAX arrays."""

    def __jax_like__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        device: jax.Device | Sharding | None = None,
        copy: bool = False,
    ) -> JaxLike[Any]:
        """Convert to a JaxLike."""


@runtime_checkable
class JaxLike[DT](ArrayLike[DT], Protocol):
    """Protocol for array-like objects that implement JAX-specific APIs.

    Every member declared here also exists on a real ``jax.Array``, otherwise
    ``isinstance(jnp.ones(3), JaxLike)`` would silently be False.

    Deliberate deviation from ``TorchLike``: ``__jax_function__`` is *not* part of this protocol.
    ``torch.Tensor`` really does implement ``__torch_function__``, but ``jax.Array`` has no
    ``__jax_function__``, so requiring it would exclude plain JAX arrays. The hook is declared on
    :class:`JaxLikeImplementation` instead. For the same reason ``mH``, ``numpy`` and ``detach``
    are dropped: JAX arrays are immutable and carry no autograd tape.
    """

    @property
    def at(self) -> Any:  # noqa: ANN401
        """Indexed update helper, see ``jax.Array.at``."""

    @property
    def sharding(self) -> Any:  # noqa: ANN401
        """The sharding of the underlying array."""

    def devices(self) -> set[jax.Device]:
        """Return the set of devices the array lives on."""

    def block_until_ready(self) -> Self:
        """Block until the asynchronous computation of the array has finished."""

    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: jax.Device | Sharding | None = None,
    ) -> Self:
        """Return a copy of the array cast to the given data type."""

    def copy(self) -> Self:
        """Return a copy of the array."""

    def reshape(self, *args: int | Sequence[int], order: str = "C") -> Self:
        """Return a reshaped version of the array."""

    def ravel(self, order: str = "C") -> Self:
        """Return a flattened version of the array."""

    def flatten(self, order: str = "C") -> Self:
        """Return a flattened version of the array."""

    def squeeze(self, axis: int | Sequence[int] | None = None) -> Self:
        """Return the array with axes of length one removed."""

    def transpose(self, *args: int | Sequence[int] | None) -> Self:
        """Return a transposed version of the array."""

    def view(self, dtype: DTypeLike | None = None, type: None = None) -> Self:  # noqa: A002
        """Return a bit-cast view of the array."""

    def item(self, *args: int) -> bool | int | float | complex:
        """Return the array as a Python scalar."""

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the array as a (nested) Python list."""


class JaxLikeImplementation[DT](ArrayLike[DT], ABC):
    """ABC implementation for array-like objects that behave like JAX arrays.

    Subclasses only have to implement ``__jax_function__`` plus the few members that cannot be
    expressed generically. Every concrete method below is written in terms of the wrappers in
    :mod:`probly.representation.jax_functions`, so it routes back into ``__jax_function__``.

    Concrete (non-abstract) subclasses are additionally registered as JAX pytree nodes, which is
    what makes ``jax.jit``, ``jax.vmap``, ``jax.grad`` and ``jax.tree`` work on them. Pytree
    registration is *not* a substitute for the override machinery: ``jnp.sum(obj)`` still raises
    for a registered pytree, which is why both mechanisms exist.

    ``at`` is deliberately absent here. It is the JAX stand-in for ``__setitem__``, and an indexed
    update has no single meaning across fields with different batch semantics (a weighted sample
    indexes its weights differently from its values; a protected-axis object must pad the index
    with its trailing axes). ``TorchLikeImplementation`` leaves ``__setitem__`` to its subclasses
    for exactly the same reason, so subclasses define ``at``.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register concrete subclasses as JAX pytree nodes.

        Args:
            **kwargs: Keyword arguments forwarded to ``super().__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)

        if cls is JaxLikeImplementation or isabstract(cls) or ABC in cls.__bases__:
            return

        # Re-registering the same class raises, which happens on module reloads.
        with suppress(ValueError):
            register_pytree_node_class(cls)

    @classmethod
    def _pytree_field_names(cls) -> tuple[str, ...]:
        """Return the names of the fields considered by the default pytree flattening."""
        if is_dataclass(cls):
            return tuple(field.name for field in fields(cls))
        return ()

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Split the object into pytree children and static auxiliary data.

        Fields holding arrays (anything exposing ``shape``, plus ``None``) become children, all
        remaining fields become auxiliary data. Subclasses may override this together with
        :meth:`tree_unflatten`.

        Returns:
            The children and the auxiliary data.
        """
        names = type(self)._pytree_field_names()  # noqa: SLF001

        children: list[Any] = []
        child_names: list[str] = []
        aux: list[tuple[str, Any]] = []

        for name in names:
            value = getattr(self, name)
            if value is None or hasattr(value, "shape"):
                child_names.append(name)
                children.append(value)
            else:
                aux.append((name, value))

        return tuple(children), (tuple(child_names), tuple(aux))

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Any, ...]) -> Self:  # noqa: ANN401
        """Rebuild an object from pytree children and auxiliary data.

        The object is built without calling ``__init__``, because JAX passes tracer objects here
        during tracing and those would fail the usual validation.

        Args:
            aux_data: The auxiliary data produced by :meth:`tree_flatten`.
            children: The children produced by :meth:`tree_flatten`.

        Returns:
            The reconstructed object.
        """
        child_names, aux = aux_data

        obj = object.__new__(cls)
        for name, value in zip(child_names, children, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in aux:
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    @abstractmethod
    def __jax_function__(
        cls,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Handle a call to one of the wrappers in :mod:`probly.representation.jax_functions`.

        Args:
            func: The wrapper function that was called.
            types: The types of the overriding arguments.
            args: The positional arguments of the call.
            kwargs: The keyword arguments of the call.

        Returns:
            The result of the call, or ``NotImplemented`` if it is not supported.
        """

    def _map_array_fields(self, op: Callable[[Any], Any]) -> Self:
        """Apply an elementwise operation to every array-valued field.

        The fields are the pytree children produced by :meth:`tree_flatten`, so a subclass that
        customises flattening automatically customises this too. ``None`` children are passed
        through untouched.

        Args:
            op: The operation to apply to each array-valued field.

        Returns:
            A copy with the operation applied to every array-valued field.
        """
        children, aux_data = self.tree_flatten()
        mapped = tuple(child if child is None else op(child) for child in children)
        return type(self).tree_unflatten(aux_data, mapped)

    def stop_gradient(self) -> Self:
        """Return a copy detached from the autodiff graph.

        This is the JAX equivalent of ``torch.Tensor.detach``: JAX arrays carry no autograd tape,
        so gradient flow is cut with ``jax.lax.stop_gradient`` instead. Subclasses holding NumPy
        sidecar fields should override this to leave those fields untouched.

        Returns:
            A copy through which gradients do not propagate.
        """
        return self._map_array_fields(jax.lax.stop_gradient)

    def block_until_ready(self) -> Self:
        """Block until the asynchronous computation of every array-valued field has finished.

        Returns:
            The object itself.
        """
        children, _ = self.tree_flatten()
        for child in children:
            if child is not None and hasattr(child, "block_until_ready"):
                child.block_until_ready()
        return self

    def astype(
        self,
        dtype: DTypeLike | None,
        copy: bool = False,
        device: jax.Device | Sharding | None = None,
    ) -> Self:
        """Return a copy of the array cast to the given data type.

        Args:
            dtype: The target data type.
            copy: Whether to always return a copy.
            device: The device the result should live on.

        Returns:
            A copy with every array-valued field cast to the given data type.
        """
        return self._map_array_fields(lambda child: jax_astype(child, dtype, copy=copy, device=device))

    @override
    def to_device(self, device: Any, /, *, stream: int | Any | None = None) -> Self:
        """Move the array to the given device.

        Args:
            device: The target device.
            stream: Unsupported by JAX, must be None.

        Returns:
            A copy with every array-valued field on the given device.

        Raises:
            NotImplementedError: If ``stream`` is not None.
        """
        if stream is not None:
            msg = "The stream argument is not supported by JAX."
            raise NotImplementedError(msg)
        return self._map_array_fields(lambda child: jax.device_put(child, device))

    @property
    def T(self) -> Self:  # noqa: N802
        """Inverts the order of the axes of the underlying array."""
        return jax_transpose(self)  # ty: ignore[invalid-return-type, invalid-argument-type]

    @property
    def mT(self) -> Self:  # noqa: N802
        """The version of the underlying array with its last two axes swapped."""
        return jax_matrix_transpose(self)  # ty: ignore[invalid-return-type, invalid-argument-type]

    @property
    def mH(self) -> Self:  # noqa: N802
        """The adjoint (conjugate) transposed version of the underlying array."""
        return jax_conj(self.mT)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def reshape(self, *args: int | Sequence[int], order: str = "C") -> Self:
        """Return a reshaped version of the array.

        Args:
            *args: The target shape, either as a single sequence or as separate integers.
            order: The read/write element order.

        Returns:
            The reshaped array.
        """
        shape = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
        return jax_reshape(self, shape, order)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def transpose(self, *args: int | Sequence[int] | None) -> Self:
        """Return a transposed version of the array.

        Args:
            *args: The permutation of the axes, either as a single sequence or as separate
                integers. If omitted, the axes are reversed.

        Returns:
            The transposed array.
        """
        if len(args) == 0:
            return jax_transpose(self)  # ty: ignore[invalid-return-type, invalid-argument-type]
        axes = args[0] if len(args) == 1 and not isinstance(args[0], int) else args
        return jax_transpose(self, axes)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def ravel(self, order: str = "C") -> Self:
        """Return a flattened version of the array.

        Args:
            order: The read/write element order.

        Returns:
            The flattened array.
        """
        return jax_reshape(self, (-1,), order)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def flatten(self, order: str = "C") -> Self:
        """Return a flattened version of the array.

        Args:
            order: The read/write element order.

        Returns:
            The flattened array.
        """
        return self.ravel(order)

    def squeeze(self, axis: int | Sequence[int] | None = None) -> Self:
        """Return the array with axes of length one removed.

        Args:
            axis: The axis or axes to remove. If omitted, all length-one axes are removed.

        Returns:
            The squeezed array.
        """
        return jax_squeeze(self, axis)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def copy(self) -> Self:
        """Return a copy of the array."""
        return jax_copy(self)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def item(self, *args: int) -> bool | int | float | complex:
        """Return the array as a Python scalar.

        Args:
            *args: Optional index of the element to return.

        Returns:
            The selected element as a Python scalar.
        """
        return np.asarray(self).item(*args)

    def tolist(self) -> Any:  # noqa: ANN401
        """Return the array as a (nested) Python list."""
        return np.asarray(self).tolist()

    @override
    def __array__(self, dtype: NpDTypeLike | None = None, /, *, copy: bool | None = None) -> NDArray[Any]:
        """Convert to a numpy array.

        Args:
            dtype: The desired data type of the output array.
            copy: Whether to return a copy of the data.

        Returns:
            The converted numpy array.
        """
        if has_jax_function((self,)):
            return handle_jax_function(np.asarray, (self,), self, dtype=dtype, copy=copy)
        return np.asarray(jnp.asarray(self), dtype=dtype, copy=copy)


JaxLikeImplementation.register(jax.Array)


@flexdispatch
def to_jax_like[DT](
    data: object,
    /,
    dtype: DTypeLike | None = None,
    *,
    device: jax.Device | Sharding | None = None,
    copy: bool = False,
) -> JaxLike[Any] | jax.Array:
    """Convert an ArrayLike to a JaxLike.

    If possible, use the __jax_like__ method to convert the array, otherwise use jax.numpy.asarray.

    Args:
        data: The data to convert.
        dtype: The desired data type of the output array.
        device: The desired device of the output array.
        copy: Whether to return a copy of the input data.

    Returns:
        The converted array.
    """
    return jnp.asarray(data, dtype=dtype, device=device, copy=copy)


@to_jax_like.register(JaxLikeConvertible)
def _[DT](
    data: JaxLikeConvertible[DT],
    /,
    dtype: DTypeLike | None = None,
    *,
    device: jax.Device | Sharding | None = None,
    copy: bool = False,
) -> JaxLike[Any] | jax.Array:
    return data.__jax_like__(dtype, device=device, copy=copy)


__all__ = ["JaxLike", "JaxLikeConvertible", "JaxLikeImplementation", "to_jax_like"]
