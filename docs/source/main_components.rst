Main components
================================
``probly`` is designed to bridge the gap between standard deterministic deep 
learning and probabilistic modeling. Instead of requiring you to rewrite your 
models from scratch to account for uncertainty, ``probly`` provides a modular 
set of tools that integrate directly into your existing PyTorch or Flax workflows.

The library is built around three core pillars that mirror the uncertainty 
estimation pipeline: **Transformation**, **Representation**, 
and **Quantification**.

1.Transformations
----------------------------
The core of ``probly`` lies in its **Transformations**. These functions take a standard, deterministic PyTorch 
(or Flax) model and automatically modify its architecture to make it uncertainty-aware.


Below are the primary transformations available in the library.

1.1 Dropout
~~~~~~~~~~~~~~~~~~~~~~~
.. TODO: Add a link to the corresponding python notebook

1.2. DropConnect 
~~~~~~~~~~~~~~~~~~~~~~~
1.3. Ensemble
~~~~~~~~~~~~~~~~~~~~~~~
1.4.Evidential
~~~~~~~~~~~~~~~~~~~~~~~
1.5.Bayesian
~~~~~~~~~~~~~~~~~~~~~~~

2. Utilities and Layers
----------------------------

3. Evaluation Tools
----------------------------