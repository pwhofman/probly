.. _contributing:

============
Contributing
============

``probly`` is in early development and contributions are welcome --- a bug
report, a missing backend for a method that already exists, or a method that is
not in the library yet.

This part of the documentation has three entry points. :ref:`contributing_guidelines`
covers the process: what we are looking for, how to set up a development
environment, and the style rules a pull request has to satisfy.
:ref:`adding_a_method` is the concrete recipe for the most common contribution,
walking a fictional method from an empty package to a gallery example. :ref:`pillars-composition`
explains the dispatch design that a new method plugs into, and why a method only
has to implement the first of the four :ref:`core pillars <core_pillars>`.

.. toctree::
    :maxdepth: 2

    guidelines
    adding_a_method
    composition
