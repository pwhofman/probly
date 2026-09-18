# Design principles

This document holds the guiding design principles for `probly`; it will be added to when new design decisions are made.

## Namespaces

Each top-level package stands for one concept, and everything in it is an instance of that concept:
`probly.method` contains methods, `probly.losses` losses, `probly.metrics` metrics. Packages are defined by what
their contents are, not by where they are used: `probly.train` contains training utilities but no losses,
although both are needed to train a model. A function that is not an instance of its intended package's concept
belongs in the package whose concept it instantiates, or in a new one.

## Backend naming

Backend-specific code is kept in a module named after the backend, `torch.py`, `jax.py`, `flax.py`, `sklearn.py` or
`array.py` for NumPy, next to a `_common.py` with the backend-free part: the dispatching function and the protocols.
A name without a backend refers to the definition in `_common.py`; every public name in a backend module includes
the name of that module, so that a dispatching function or another backend can be added without renaming anything
and no two backend modules define the same name. NumPy is called `array` throughout: `array.py`, `array_accuracy`,
`ArraySample`.

The backend is a prefix, for functions and for classes: `torch_entropy`, `TorchSample`. When a module has several
implementations of one function, one per representation, the representation is named between the backend and the
function: `torch_categorical_entropy` and `torch_dirichlet_entropy` are both implementations of `entropy`.
