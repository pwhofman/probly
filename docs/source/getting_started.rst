.. _getting_started:
Getting Started
================

``probly`` is an open source library giving easy access to uncertainty representation and quantification by allowing
the transformation of any PyTorch, Flax/JAX or scikit-learn model into an uncertainty aware one with a single line of code.
It offers an extensive collection of methods, representations as well as their corresponding quantification measures.
``probly``aims to to unify what already exists in one easy to use implementation without hard dependencies to only one
framework or particular approach to modeling uncertainty.
Just chose your preferred method and let ``probly`` do the rest for you.

This section illustrates the basic pipeline based on the four pillars: Transformation, Representation, Quantification and Evaluation.
It serves as an easy to understand overview.
For your next steps please refer to our :ref:`installation` instructions. For a more detailed explanation on the application
of ``probly`` and what methods to chose skip to :ref:`next_steps`.

Pipeline
---------
``probly`` makes it especially easy to make models uncertainty-aware by applying several downstream tasks automatically.

Transformation
~~~~~~~~~~~~~~~
At the core of ``probly`` lay the transformations.

.. _next_steps:
Next Steps
-----------
Most of the pipeline happens behind the curtain, all of it being covered and automated with just one line code.
This shows the easy application of ``probly``! The actual power and difficulty lies in choosing the right method. To get suggestions
on how to proceed and find out how to best use ``probly`` for your own applications refer to :ref:`user_guide`.

For the exhaustive overview of methods check out :ref:`api_ref` and for a practical insight look at :ref:`examples`.
Found a method that is missing please refer to our guide :ref:`adding_a_method`.

Finally have fun using ``probly`!
