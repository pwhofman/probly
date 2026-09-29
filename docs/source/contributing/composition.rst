.. _pillars-composition:

======================
Why the Stages Compose
======================

:ref:`core_pillars` splits the library into transformation, representation,
quantification, and evaluation. Three dispatch mechanisms hold those pillars
apart:

- **Type dispatch** (``flexdispatch``) routes ``predict``, ``representer``, and
  ``quantify`` to the right backend based on the object handed in.
- **Traverser dispatch** (``flexdispatch_traverser``) walks a network layer by
  layer, so a transformation is defined per layer type instead of per model.
- **Value dispatch** (``switchdispatch``) maps names to implementations, which
  is what makes string arguments such as ``predictor_type="logit_classifier"``
  work.

The practical consequence is the benchmark: one pipeline, 20+ methods, changed
one line at a time. If you are adding a method, the pillar structure is also the
checklist --- a new method needs a transformation, must declare which
representation it produces, and inherits stages 3 and 4 for free. That is why
the recipe in :ref:`adding_a_method` is mostly about pillar 1: registering the
per-layer transformations for each backend, and then declaring the hooks that
tell pillars 2 and 3 what the method hands back.

.. seealso::

    :ref:`adding_a_method` for the concrete checklist, and :ref:`methods` for the
    catalogue, grouped by the representation each method produces.
