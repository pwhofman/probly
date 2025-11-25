Advanced Topics
===============

1. Overview
-----------

1.1 Purpose of this chapter
This chapter explains:

- what “advanced” means in the context of `probly`,
- when you should read this chapter (recommended after Core Concepts and Main Components).

1.2 Prerequisites & notatin
Before reading this chapter, the reader should already be familiar with:

- the concepts introduced in Core Concepts,  
- the basic workflows described in Main Components,  
- foundational ideas such as uncertainty representations, transformations, and inference.

For clarity, this chapter follows the same notation conventions used throughout the Probly documentation.

1.3 Typical advanced use cases
This chapter is intended for scenarios where users go beyond simple toy examples, such as:

- training or evaluating large or real-world models,  
- dealing with tight performance or memory constraints,  
- integrating Probly into existing machine-learning pipelines.

These use cases often require a deeper understanding of transformations, scalability, and framework interoperability, which this chapter provides.

.. seealso::

    For background material, see :doc:`Core Concepts <core_concepts>`.

    For the main bulding blocks of `probly`, like the main transofrmations, utilities & layers, and evaluation tools, see :doc:`Main Components <main_components>`.

2. Custom Transformations
-------------------------

2.1 Recall: What is a transformation?

In *probly*, a **transformation** is a small building block that maps values between two spaces:

- an **unconstrained space**, where optimisation and inference algorithms can work freely, and  
- a **constrained space**, which matches the natural domain of your parameters or predictions
  (for example positive scales, probabilities on a simplex, or bounded intervals).

Instead of forcing you to design models directly in a complicated constrained space, you write
your model in terms of meaningful parameters. The transformation then takes care of the math that
keeps everything inside the valid domain.

In practice this means that transformations:

- provide a *short, reusable recipe* for how to turn raw latent variables into valid parameters,
- enable **reparameterisation**, which can make optimisation easier and gradients better behaved,
- automatically enforce **constraints** such as positivity, bounds, or simplex structure.

You can think of a transformation as an adapter between “nice for the optimiser” coordinates and
“nice for the human” coordinates.
