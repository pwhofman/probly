.. _contributing_guidelines:

=======================
Contributing Guidelines
=======================

The authoritative guidelines live in the repository, in
`.github/CONTRIBUTING.md <https://github.com/pwhofman/probly/blob/main/.github/CONTRIBUTING.md>`_.
Read them before opening a pull request; this page only summarizes what is in
them so you know whether you need to.

:What to work on: Representation, quantification, and calibration methods,
    downstream tasks, datasets and dataloaders, and ports of existing features
    to PyTorch, HuggingFace, and scikit-learn. For anything outside that scope,
    open an issue first so we can discuss it.
:Workflow: Fork ``main``, clone the fork, commit, push, and open a pull request
    against ``main``.
:Development environment: Python 3.12+ with `uv <https://docs.astral.sh/uv/>`_,
    then ``uv sync --dev``. ``probly`` itself supports Python 3.12+, but the
    ``dev`` and ``docs`` dependency groups need Python 3.13+, which is what
    ``.python-version`` pins the development environment to.
:Code style: `Ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting,
    configured in ``pyproject.toml`` and applied by the pre-commit hook.
:Documentation: Public features are documented in their docstring, following the
    `Google style guide <https://google.github.io/styleguide/pyguide.html#docstrings>`_.
    If the feature comes from a paper, cite it.
:Credit: If you use code from another source, check its license and credit the
    original authors.

.. seealso::

    :ref:`adding_a_method` for the file layout, registration mechanics, and
    quality checks that a new uncertainty method needs.
