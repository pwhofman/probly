.. _jax_representations:

===================
JAX Representations
===================

This page covers the machinery behind ``probly``'s JAX representation types:
how Python operators on protected-axis representations are dispatched and
permitted, and how constructor validation interacts with ``jax.jit``. It is
mostly relevant when implementing a new JAX representation, or when compiling
code that constructs one.

Protected-axis operators
========================

``JaxAxisProtected`` provides Python array operators through
``JaxOperatorsMixin``. Operators dispatch through probly's ``__jax_function__``
protocol, using the same wrapper functions as explicit function calls. For
example, ``x + y`` and ``jax_add(x, y)`` share one implementation and permission.
Native ``jax.numpy`` calls do not use this override protocol.

Wrapper implementations use ``jax_function_dispatch`` to declare which
parameters participate: ``@jax_function_dispatch("x1", "x2")`` selects two
operands, while ``@jax_function_dispatch(unpack=("arrays",))`` selects sequence
elements. The decorator preserves signatures and docstrings and dispatches
using the exported wrapper as the function key. Operator overloads give native
calls precise array result types (a pair of arrays for ``jax_divmod``).
Custom operands implement ``SupportsJaxFunction``; their results are typed as
``object`` because the override protocol does not guarantee a particular
representation or result type.

Enabling operations
-------------------

Arithmetic is disabled by default. A representation opts in by adding wrapper
functions to its ``permitted_functions`` set:

.. jupyter-execute::

    from collections.abc import Callable
    from dataclasses import dataclass
    from typing import ClassVar

    import jax
    import jax.numpy as jnp

    from ndlike.protected_axis.jax import JaxAxisProtected
    from ndlike.jax_functions import jax_add, jax_multiply

    @dataclass(frozen=True, eq=False)
    class Vectors(JaxAxisProtected[jax.Array]):
        array: jax.Array
        protected_axes: ClassVar[dict[str, int]] = {"array": 1}
        permitted_functions: ClassVar[set[Callable]] = {jax_add, jax_multiply}

    x = Vectors(jnp.ones((2, 3)))
    y = 2 * x + x
    assert y.shape == (2,)
    assert y.protected_shape == (3,)

Both protected operands must permit the operation. Forward and reflected
operators share a permission; permitting addition does not permit multiplication
or a reduction such as ``jax_sum``. Unsupported binary operations return
``NotImplemented`` to Python, while explicit wrapper calls raise ``TypeError``
if every implementation declines.

Supported semantics
-------------------

* Arithmetic, bitwise operations, shifts, and unary operators apply fieldwise.
  Protected operands must have the same concrete type, protected-axis layout,
  and protected trailing shapes. Their visible batch dimensions may broadcast.
* Ordinary numeric scalars and arrays apply to every field using field-level
  broadcasting. An ordinary array is not automatically interpreted as batch
  weights or expanded with trailing singleton axes. Operations that change
  protected shapes or produce inconsistent batch shapes are rejected.
* Comparisons return boolean arrays with the visible batch shape. All protected
  components and fields must satisfy the relation; ``!=`` instead tests whether
  any component differs. Custom subclass comparisons take precedence.
  ``JaxLikeImplementation`` preserves inherited array equality even when a
  subclass uses the default dataclass equality setting.
  When equality dispatch declines, Python retains its identity fallback.
* ``divmod`` returns two reconstructed representations.
* ``@`` applies matrix multiplication to visible dimensions independently at
  each protected coordinate. Ordinary array operands use their normal matrix
  dimensions. Vector and batched matrix multiplication follow JAX semantics.
* Augmented assignment, such as ``x += y``, rebinds an out-of-place result.
  Modular three-argument ``pow`` is unsupported.
* Generic JAX operators reject representations with NumPy sidecar fields.
  NumPy ufuncs may coerce through ``__array__``, losing the representation and
  its JAX-specific operation permissions. Use the JAX wrappers when those
  semantics must be preserved.

Distribution-specific behavior
------------------------------

Protected-axis storage alone does not define distribution arithmetic.
``JaxDirichletDistribution`` permits addition and subtraction and clamps the
resulting concentrations to a positive minimum. ``JaxGaussianDistribution``
permits addition and subtraction: constants shift the mean while preserving
variance, and two Gaussian operands are treated as independent, so their
variances add for both sums and differences.

Subclasses can customize arithmetic results through
``_postprocess_elementwise_result(values, *, func, operands)``. The hook receives
the wrapper function and original operands in expression order. Its results
are checked for protected-shape and batch-shape preservation before
``with_protected_values(values, func)`` reconstructs the representation.

Sample operators and pytree fields
=================================

``JaxSample`` also dispatches elementwise operators through these wrappers.
Broadcasting tracks the sample axis and retains weights; conflicting sample
axes produce a native result, and conflicting weights raise a validation error.
For compiled operations on two distinct weighted samples, use ``checkify`` to
functionalize the weight-compatibility check, as for constructor checks below.
Native JAX arrays on the left of an infix operation may consume
``__jax_array__`` and return native arrays; explicit ndlike wrappers preserve
sample metadata in either operand order.
Samples use NumPy priority zero: NumPy-led infix calls and explicit NumPy
ufuncs can return native NumPy arrays. This does not affect the explicit
``__jax_function__`` dispatch used by ndlike's JAX wrappers. JAX also accepts
NumPy arrays as inputs, so sample-led JAX operations can retain the sample
wrapper even with a NumPy array on the right.

Reducing the sample axis removes the sample wrapper, even with
``keepdims=True``. Keeping a length-one dimension preserves shape, not the
sample interpretation. Reductions over other axes retain the wrapper and
weights, updating the sample axis only if dimensions before it are removed.

Since ``ndlike`` 0.2, general ``JaxLikeImplementation`` subclasses explicitly
declare dynamic fields. ``JaxSample`` declares
``pytree_children = ("array", "weights")``, including ``weights`` when its value
is ``None``. The sample axis remains static metadata. Subclasses adding dynamic
fields must extend this tuple. ``JaxAxisProtected`` declares its protected fields
automatically.

Native ``jax.tree.map`` transforms the underlying array and weights structurally.
Use ``ndlike.jax_like.jax_tree_map`` to apply operations to whole sample objects
instead: multiplying a sample by two transforms its values but not its weights.

Constructor validation
======================

JAX representation types use
``jax.experimental.checkify.check`` for their constructor argument value validation
in ``__post_init__``. Type and shape validation uses ordinary Python checks.
Boolean-backed conformal sets need no binary-value check.

Eager construction still raises a ``ValueError`` subclass for invalid values.
When a compiled function constructs or reconstructs one of these representations
(including through indexing, reshaping, arithmetic, or sampling), functionalize
the checks with ``checkify``:

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify

    from probly.representation.distribution.jax_categorical import (
        JaxProbabilityCategoricalDistribution,
    )
    from ndlike.jax_functions import jax_mean

    distribution = JaxProbabilityCategoricalDistribution(
        jnp.array([[0.2, 0.8], [0.4, 0.6]])
    )
    checked_mean = jax.jit(checkify.checkify(lambda d: jax_mean(d, axis=0)))
    error, result = checked_mean(distribution)
    error.throw()  # Check the error outside the compiled function.
    print(result)

Plain ``jax.jit`` cannot stage these checks without ``checkify``. Checks run on
each execution, including calls that reuse compiled code. Checkified functions
also compose with ``jax.vmap`` and ``jax.grad``. Operations that only read
parameters and do not invoke validation can still use plain ``jax.jit``.
