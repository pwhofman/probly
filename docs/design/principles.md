# Design principles

This document holds the guiding design principles for `probly`; it will be added to when new design decisions are made.

## Namespaces

Each top-level package stands for one concept, and everything in it is an instance of that concept:
`probly.method` contains methods, `probly.losses` losses, `probly.metrics` metrics. Packages are defined by what
their contents are, not by where they are used: `probly.train` contains training utilities but no losses,
although both are needed to train a model. A function that is not an instance of its intended package's concept
belongs in the package whose concept it instantiates, or in a new one.
