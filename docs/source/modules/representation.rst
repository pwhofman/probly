.. _representation:

==============
Representation
==============

.. currentmodule:: probly.representation

Categorical score targets
=========================

The inner-product, KL-divergence, total-variation, and Wasserstein conformal
scores interpret targets consistently across NumPy, Torch, and JAX:

* Integer arrays contain class indices, interpreted as point-mass targets.
* Floating-point arrays contain probability vectors with a trailing class axis.
* ``CategoricalDistribution`` targets supply their normalized probabilities,
  regardless of their storage dtype or whether they store logits.
* Boolean and complex target arrays are rejected.

Types and dtypes determine the interpretation. Shapes only validate the class
count and establish ordinary batch broadcasting. For example, predictions of
shape ``(members, batch, classes)`` accept floating targets of shape
``(batch, classes)`` or integer labels of shape ``(batch,)``. Singleton batch
dimensions are preserved according to normal broadcasting rules.

Integer one-hot arrays must be cast to floating point or wrapped in a
categorical distribution to be interpreted as probability vectors. Conversely,
class labels stored as floats must be explicitly converted to an integer dtype.
There is no special reshaping of row-vector labels and no rank-based inference
of target semantics.

JAX protected-axis operators
============================

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

.. code-block:: python

    from collections.abc import Callable
    from dataclasses import dataclass
    from typing import ClassVar

    import jax
    import jax.numpy as jnp

    from probly.representation._protected_axis.jax import JaxAxisProtected
    from probly.representation.jax_functions import jax_add, jax_multiply

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
  any component differs. Custom subclass comparisons take precedence. Use
  ``eq=False`` on dataclasses to inherit the mixin's equality implementation.
  When equality dispatch declines, Python retains its identity fallback.
* ``divmod`` returns two reconstructed representations.
* ``@`` applies matrix multiplication to visible dimensions independently at
  each protected coordinate. Ordinary array operands use their normal matrix
  dimensions. Vector and batched matrix multiplication follow JAX semantics.
* Augmented assignment, such as ``x += y``, rebinds an out-of-place result.
  Modular three-argument ``pow`` is unsupported.
* Generic operators reject representations with NumPy sidecar fields. NumPy
  ufunc coercion is disabled so it cannot bypass the JAX function permissions;
  explicit conversion with ``np.asarray`` remains available.

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

JAX constructor validation
--------------------------

JAX representation types use
``jax.experimental.checkify.check`` for their constructor argument value validation
in ``__post_init__``. Type and shape validation uses ordinary Python checks.
Boolean-backed conformal sets need no binary-value check.

Eager construction still raises a ``ValueError`` subclass for invalid values.
When a compiled function constructs or reconstructs one of these representations
(including through indexing, reshaping, arithmetic, or sampling), functionalize
the checks with ``checkify``:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from jax.experimental import checkify

    from probly.representation.distribution.jax_categorical import (
        JaxProbabilityCategoricalDistribution,
    )
    from probly.representation.jax_functions import jax_mean

    distribution = JaxProbabilityCategoricalDistribution(
        jnp.array([[0.2, 0.8], [0.4, 0.6]])
    )
    checked_mean = jax.jit(checkify.checkify(lambda d: jax_mean(d, axis=0)))
    error, result = checked_mean(distribution)
    error.throw()  # Check the error outside the compiled function.

Plain ``jax.jit`` cannot stage these checks without ``checkify``. Checks run on
each execution, including calls that reuse compiled code. Checkified functions
also compose with ``jax.vmap`` and ``jax.grad``. Operations that only read
parameters and do not invoke validation can still use plain ``jax.jit``.
