.. _nn-init-doc:
.. _locally-disable-grad-doc:
.. _index:

``probly``
==========

Welcome to the documentation for ``probly``, a Python library for uncertainty quantification in
machine learning. ``probly``'s name, coming from *probably this is the right answer*, reflects
its **core functionality**: Providing an easy-to-use interface for allowing practitioners to incorporate
uncertainty into their machine learning workflows.

.. admonition:: ``probly``'s Philosophy

    1. **Library-agnostic**: ``probly`` is designed to work with any machine learning framework, that you can use it with. Let it be `PyTorch <https://pytorch.org>`_, `Flax <https://flax.readthedocs.io/en/latest/>`_, `JAX <https://jax.readthedocs.io/en/latest/>`_, `TensorFlow <https://www.tensorflow.org>`_, `scikit-learn <https://scikit-learn.org/stable/>`_, and more.
    2. **Model-agnostic**: ``probly`` is designed to work with any machine learning model and pipelines that you are already using: from simple linear regression to complex transformer-based models, and anything in between.
    3. **Ante-Hoc and Post-Hoc**: You either bring your own model and ``probly`` will transform it or you can use the built-in tools to build your own models with uncertainty directly in mind.



.. toctree::
   :maxdepth: 1
   :caption: Table of Contents

   introduction
   installation
   user_guide
   examples
   api
   references
   faq
