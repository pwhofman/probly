.. _adding_a_method:

===================
Adding a New Method
===================

This guide walks through adding a new uncertainty quantification method to ``probly``,
step by step. It complements the general
`contributing guidelines <https://github.com/pwhofman/probly/blob/main/.github/CONTRIBUTING.md>`_
with the concrete file layout, registration mechanics, and quality checks that a new
method needs. As a running example we scaffold a fictional method called ``mymethod``.

What is a method?
=================

In ``probly``, a *method* is a predictor transformation: a function that takes a base
model (a PyTorch module, a Flax module, a scikit-learn estimator, ...) and returns an
uncertainty-aware predictor. Examples are :func:`~probly.method.dropout`,
:func:`~probly.method.sngp`, and :func:`~probly.method.vbll`. The transformed predictor
then plugs into the rest of the pipeline: *representation* (turning stochastic
predictions into, e.g., a sample of probability vectors), *quantification* (computing
uncertainty measures from the representation), and *evaluation*.

``probly`` is backend-agnostic. A method is therefore split into a backend-agnostic
core and one thin implementation module per supported backend, wired together through
lazy dispatch so that no backend is imported before it is actually used.

Anatomy of a method
===================

A new method lives in its own package under ``src/probly/method/``:

.. code-block:: text

    src/probly/method/mymethod/
        __init__.py     # public exports + lazy backend registration
        _common.py      # backend-agnostic core: protocol, traverser, transformation
        torch.py        # PyTorch implementation
        flax.py         # Flax implementation (optional, one file per backend)

Depending on what the method needs, it may also touch:

.. code-block:: text

    src/probly/layers/torch.py            # reusable custom layers (e.g. SNGPLayer)
    src/probly/train/mymethod/            # custom training losses / loops (e.g. vbll)
    src/probly/method/__init__.py         # export the new transformation
    docs/source/references.bib            # BibTeX entry for the paper
    examples/method/plot_mymethod.py      # gallery example
    tests/probly/method/mymethod/         # tests, split by backend

Composing from existing building blocks
---------------------------------------

Not every method needs its own traverser and layer transformations. ``probly``
already ships many base transformations in ``src/probly/transformation/`` --
``ensemble``, ``subensemble``, ``dropout``, ``dropconnect``, ``batchensemble``,
``bayesian``, ``calibration``, and more -- and a new method can often be composed
from them. In that case your transformation function simply calls the existing
building blocks and adds what is specific to your method, such as a predictor
protocol and a custom representer or decomposition.

For example, :func:`~probly.method.credal_ensembling` reuses
``probly.transformation.ensemble.ensemble`` to replicate the base model and only
adds a protocol that routes prediction through a credal set representer, and
:func:`~probly.method.dare` builds on ``ensemble`` and the ``subensemble``
generator. If your method boils down to "existing transformation + different
representation or quantification", start there; you may be able to skip Steps 1-3
below almost entirely and only write the composition, hooks, tests, and docs.

Step 1: Write the backend-agnostic core (``_common.py``)
========================================================

The core module defines everything that does not depend on a specific backend:

1. **A predictor protocol** that marks predictors produced by your method. If the
   transformed model is stochastic at prediction time (like MC dropout), inherit from
   ``probly.predictor.RandomPredictor``; otherwise use ``probly.predictor.Predictor``.

2. **Global variables** for the parameters your layer transformations need. These are
   passed through the traversal instead of function arguments.

3. **A traverser** created with ``flexdispatch_traverser``. It walks the layer tree of
   the base model and dispatches a transformation per layer type. Backends register
   their handlers with it (Step 2).

4. **The transformation function** itself, decorated with
   ``@predictor_transformation`` (which validates the input and infers the backend) and
   ``@MyMethodPredictor.register_factory`` (which marks the return value as an instance
   of your protocol).

.. code-block:: python

    """Shared mymethod implementation."""

    from __future__ import annotations

    from typing import TYPE_CHECKING, Protocol, runtime_checkable

    from probly.predictor import RandomPredictor
    from probly.transformation.transformation import predictor_transformation
    from probly.traverse_nn import nn_compose
    from pytraverse import CLONE, GlobalVariable, flexdispatch_traverser, traverse

    if TYPE_CHECKING:
        from flextype.isinstance import LazyType

        from probly.predictor import Predictor
        from pytraverse.composition import RegisteredLooseTraverser


    @runtime_checkable
    class MyMethodPredictor[**In, Out](RandomPredictor[In, Out], Protocol):
        """A predictor transformed by mymethod."""


    STRENGTH = GlobalVariable[float]("STRENGTH", "The strength of the perturbation.")

    mymethod_traverser = flexdispatch_traverser[object](name="mymethod_traverser")


    def register(cls: LazyType, traverser: RegisteredLooseTraverser) -> None:
        """Register a layer class to be transformed by mymethod."""
        mymethod_traverser.register(cls=cls, traverser=traverser, vars={"strength": STRENGTH})


    @predictor_transformation(permitted_predictor_types=None, preserve_predictor_type=True)
    @MyMethodPredictor.register_factory
    def mymethod[T: Predictor](base: T, strength: float = 0.1) -> T:
        """Create a mymethod predictor from a base predictor based on :cite:`authorPaper2026`.

        Args:
            base: The base model to transform.
            strength: The strength of the perturbation. Default is 0.1.

        Returns:
            The mymethod predictor.
        """
        return traverse(
            base,
            nn_compose(mymethod_traverser),
            init={STRENGTH: strength, CLONE: True},
        )

A few notes on the decorators:

* ``predictor_transformation(permitted_predictor_types=..., preserve_predictor_type=...)``
  controls which predictor types the transformation accepts and whether the returned
  object keeps the type of the input (``dropout`` preserves it, ``sngp`` does not
  because its output type changes to a distribution).
* The ``register`` helper is optional but recommended: it fixes the traverser options
  (skip rules, variable mapping) in one place so backend modules only supply the
  per-layer transformation.
* If your method is based on a paper, cite it with ``:cite:`...``` in the docstring
  (see Step 5).

Prediction, representation, and quantification hooks
----------------------------------------------------

How much more you need in ``_common.py`` depends on what your method outputs:

* **Stochastic forward passes** (dropout-style): nothing more is needed. Because your
  protocol inherits from ``RandomPredictor``, the generic
  ``probly.representer.sampler.Sampler`` representer already knows how to draw
  repeated predictions and build a sample representation, and the standard
  sample-based quantification applies.

* **Distributional outputs** (SNGP-style, the model returns e.g. a Gaussian over
  logits): register a custom ``predict`` implementation and, if needed, a custom
  representer and decomposition:

  .. code-block:: python

      from probly.predictor import predict, predict_raw


      @predict.register(MyMethodPredictor)
      def _[**In](predictor: MyMethodPredictor[In, Out], *args: In.args, **kwargs: In.kwargs) -> Out:
          """Predict method for mymethod predictors."""
          return some_distribution_from(predict_raw(predictor, *args, **kwargs))

  See ``src/probly/method/sngp/_common.py`` for a complete example that registers a
  ``Representer`` with ``@representer.register(SNGPPredictor)`` and a custom
  uncertainty decomposition with ``@decompose.register(...)``.

* **Wrapper predictors** (the method returns a new predictor class holding the base model,
  like ``swag``): register a ``predict_raw`` implementation that routes the call through
  ``predict_raw`` of the wrapped model, adding whatever context makes the inner call
  equivalent to the wrapper's forward (e.g. loading sampled weights). This keeps wrappers
  transparent to integrations that adapt model outputs, such as the transformers binding.
  See ``src/probly/method/swag/torch.py`` and the ensemble ``nn.ModuleList`` registration
  for the pattern.

Step 2: Implement the backends (``torch.py``, ``flax.py``, ...)
===============================================================

Each backend module imports its framework at module level (this is safe because the
module is only imported lazily, see Step 3), defines the per-layer transformation, and
registers it.

Use backend prefixes for implementation names: ``numpy_``, ``jax_``, ``torch_``,
``flax_``, or ``sklearn_`` for functions, and ``Numpy``, ``Jax``, ``Torch``, ``Flax``,
or ``Sklearn`` for classes. Private names retain their leading underscore, as in
``_torch_transform_linear``. Conversion names such as ``from_numpy_sample``
describe their inputs and retain that ordering.

Backend checks
--------------

The ``BKN001`` check in ``scripts/check_backend_naming.py`` enforces backend-word
placement in function, method, nested function, class, and ``type`` alias
definitions under ``src/probly``. The same script checks backend import placement
with ``BKN002``. Both rules run automatically through pre-commit and CI. To run
them directly:

.. code-block:: bash

    uv run python scripts/check_backend_naming.py

The checker recognizes exactly ``torch``, ``jax``, ``flax``, ``numpy``, ``Torch``,
``Jax``, ``Flax``, and ``Numpy`` as complete words separated by underscores or
CamelCase boundaries, including acronym boundaries such as ``HTTP|Torch``.
Digits stay within words. A backend word must be the first word, optionally
following one leading underscore. Once a backend prefix is present, additional
backend words are allowed: ``torch_numpy_predict`` and ``TorchNumpyAdapter`` both
pass. Names such as ``predict_torch`` and ``MyTorchModel`` fail, while
``convert_pytorch`` and ``MyFlaxifyModel`` pass because the backend spellings are
parts of larger words. Names without a recognized backend word are permitted.

Semantic first words ``from``, ``to``, ``is``, ``has``, ``supports``, and
``Supports`` are also permitted, using the same underscore/CamelCase boundaries
and optional privacy marker. For example, ``supports_torch`` and ``SupportsTorch``
both pass, while ``SupportsomethingTorch`` fails. Dunder
protocol functions/methods such as ``__torch_function__`` are exempt. Variables,
parameters, and imported identifiers are outside BKN001's scope. Tests are outside
both rules' default scope. Explicit file or directory arguments can be supplied
for a manual check.

For an intentional exception, add ``# noqa: BKN001`` with a reason on the opening
line containing ``def``, ``class``, or ``type``. This suppresses only that
definition, including when its signature spans multiple lines. A bare ``# noqa``
does not suppress this check.

**Backend imports (BKN002).** Runtime ``import`` and ``from ... import ...``
statements must be in appropriately named modules:

.. list-table:: Allowed module prefixes
   :header-rows: 1

   * - Imported package
     - Module prefixes
   * - ``torch``
     - ``torch``, ``transformers``, ``huggingface``, ``peft``
   * - ``jax``
     - ``jax``, ``flax``
   * - ``flax``
     - ``flax``

The rule checks the actual imported package, including submodules such as
``torch.nn`` or ``jax.numpy``, independently of import aliases. The first word of
the filename stem determines the module prefix, using the same word boundaries
and optional leading underscore as BKN001. For ``__init__.py``, the containing
package's name is used. Thus ``torch_metrics.py`` and ``torch/__init__.py`` permit
Torch imports; ``metrics_torch.py`` and ``torch/shared.py`` do not. An enclosing
backend package does not exempt a differently named module.

Function-local imports and imports inside ``try/except`` blocks are checked.
Type-only branches guarded by ``typing.TYPE_CHECKING`` or an explicitly imported
``TYPE_CHECKING`` are exempt, including import aliases and equivalent
``typing_extensions`` guards. Negated guards are supported, and runtime branches
remain checked. Relative project imports such as ``from . import torch`` are
not imports of the external Torch package. This is a check of Python import
statements, not a transitive dependency or dynamic-import analysis.

An intentional runtime bridge can use ``# noqa: BKN002`` with a reason on the
import's opening line. Suppressions are rule-specific: BKN001 does not suppress
BKN002, or vice versa. Prefer a backend-specific module and ``delayed_register``
for backend implementations used by a shared API.

Example backend implementation:

.. code-block:: python

    """Torch mymethod implementation."""

    from __future__ import annotations

    from torch import nn

    from ._common import register


    def torch_transform_linear(obj: nn.Linear, strength: float) -> nn.Module:
        """Replace a Linear layer with its mymethod counterpart."""
        return nn.Sequential(MyMethodLayer(strength=strength), obj)


    register(nn.Linear, torch_transform_linear)

The keyword arguments of the transformation function (here ``strength``) are filled
from the global variables declared in the ``register`` helper's ``vars`` mapping.

If your method needs a custom layer (a new ``nn.Module``), put it in
``src/probly/layers/torch.py`` so it can be reused and tested independently. If it
needs a special training loss or loop, add a ``src/probly/train/mymethod/`` package
(see ``src/probly/train/vbll/`` for an example).

Step 3: Wire up lazy registration (``__init__.py``)
===================================================

Backends must not be imported unless the user actually passes a model of that backend.
This is achieved with ``delayed_register`` and the fully-qualified type strings from
``probly/lazy_types.py``: the callback runs the first time the traverser encounters an
object of that type, and importing the backend module executes its ``register(...)``
calls.

.. code-block:: python

    """Mymethod implementation for uncertainty quantification."""

    from __future__ import annotations

    from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

    from ._common import MyMethodPredictor, mymethod, mymethod_traverser, register


    ## Torch
    @mymethod_traverser.delayed_register(TORCH_MODULE)
    def _(_: type) -> None:
        from . import torch as torch  # noqa: PLC0415


    ## Flax
    @mymethod_traverser.delayed_register(FLAX_MODULE)
    def _(_: type) -> None:
        from . import flax as flax  # noqa: PLC0415


    __all__ = [
        "MyMethodPredictor",
        "mymethod",
        "mymethod_traverser",
        "register",
    ]

If ``_common.py`` defines additional ``flexdispatch`` functions with backend-specific
implementations (e.g. converting a tensor sample), give them their own
``delayed_register`` hooks keyed on the appropriate lazy types (``TORCH_TENSOR``,
``JAX_ARRAY``, ...); see ``src/probly/method/sngp/__init__.py``.

Step 4: Export the method
=========================

Add the transformation to ``src/probly/method/__init__.py`` (import and ``__all__``,
both alphabetically sorted):

.. code-block:: python

    from probly.method.mymethod import mymethod

Users can now call ``probly.method.mymethod(model)``.

Step 5: Add the reference
=========================

Add the paper's BibTeX entry to ``docs/source/references.bib`` and cite it in the
transformation's docstring with ``:cite:`authorPaper2026```. The citation then renders
as a link into the :ref:`references` page of the documentation.

Step 6: Write tests
===================

Tests live under ``tests/probly/method/mymethod/`` (do not forget the ``__init__.py``)
and are split by backend:

* ``test_common.py`` for backend-agnostic checks,
* ``test_torch.py``, ``test_flax.py``, ``test_numpy.py``, ``test_jax.py``, ... for backend-specific
  checks.

Backend-specific test files call ``pytest.importorskip`` once at the top instead of
per-test skip decorators:

.. code-block:: python

    """Tests for the torch mymethod implementation."""

    from __future__ import annotations

    import pytest

    from probly.method.mymethod import mymethod
    from probly.predictor import predict

    torch = pytest.importorskip("torch")

    from torch import nn  # noqa: E402


    def test_mymethod_transforms_linear_layers() -> None:
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
        predictor = mymethod(model, strength=0.2)

        out = predict(predictor, torch.ones(4, 10))

        assert out.shape == (4, 3)

Good things to cover: the layer tree is transformed as expected, the original model is
not mutated (``CLONE: True``), parameters validate their ranges, prediction shapes are
correct, and the predictor works with the downstream representation and quantification
steps (see ``tests/probly/method/vbll/test_torch.py`` for a full example).

Run the tests with:

.. code-block:: bash

    uv run pytest tests/probly/method/mymethod

Step 7: Add a gallery example
=============================

Add a runnable example ``examples/method/plot_mymethod.py``. Examples are rendered
into the documentation by sphinx-gallery; they start with an rST docstring header and
should produce a plot:

.. code-block:: python

    """========
    Mymethod
    ========

    This example demonstrates mymethod on a toy classification task.
    """

    # %%
    # Transform the model
    # -------------------
    # ...

Keep the example small and fast; heavier variants (e.g. on MNIST) go in a separate
``plot_mymethod_mnist.py`` file, following the existing examples in
``examples/method/``.

The gallery seeds ``random``, ``numpy`` and ``torch`` before every example, so an
example that trains a model gets the same result on every build and cannot fail the
docs build only some of the time. Call ``torch.manual_seed`` yourself only when the
example needs a specific seed; it then takes precedence.

Step 8: Quality checks
======================

Before opening a pull request:

.. code-block:: bash

    # lint + format (pre-commit hooks)
    uv run prek run --all-files

    # type checking
    uv run ty check src/probly/method/mymethod

    # tests
    uv run pytest tests/probly/method/mymethod

    # docs build (incremental; renders the new example and API pages)
    uv run sphinx-build -j auto -b html docs/source docs/build/html

Checklist
=========

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Item
     - Location
   * - Backend-agnostic core (protocol, traverser, transformation)
     - ``src/probly/method/mymethod/_common.py``
   * - Backend implementations
     - ``src/probly/method/mymethod/torch.py``, ``flax.py``, ...
   * - Lazy backend registration and exports
     - ``src/probly/method/mymethod/__init__.py``
   * - Top-level export
     - ``src/probly/method/__init__.py``
   * - Custom layers (if any)
     - ``src/probly/layers/torch.py``
   * - Custom training utilities (if any)
     - ``src/probly/train/mymethod/``
   * - Paper reference
     - ``docs/source/references.bib`` + ``:cite:`` in the docstring
   * - Tests (split by backend)
     - ``tests/probly/method/mymethod/``
   * - Gallery example
     - ``examples/method/plot_mymethod.py``
