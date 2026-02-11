=========
Changelog
=========

[2026-01-20] - Documentation Format
==============================================

Documentation Converted from Markdown to reStructuredText
--------------------------------------------------------

**Main Changes:**

- All documentation files converted from Markdown (.md) to reStructuredText (.rst)
- Code blocks updated with ``.. code-block::`` directive
- Tables converted to ``list-table`` format
- Inline code changed to double backticks (````code````)
- Heading hierarchy adjusted to RST standard
- File references updated from .md to .rst

**Converted Files:**

- ``README.md`` → ``README.rst``
- ``docs/data_generation_guide.md`` → ``docs/data_generation_guide.rst``
- ``docs/api_reference.md`` → ``docs/api_reference.rst``
- ``docs/multi_framework_guide.md`` → ``docs/multi_framework_guide.rst``

**Preserved Files:**

- ``examples/first_order_tutorial.ipynb`` - Jupyter Notebook (no changes)
- ``examples/simple_usage.py`` - Python example script (no changes)


[2026-01-11] - Original Documentation
===========================================

Documentation Created and Updated
----------------------------------------

**Main Changes:**

- README consolidated (2 files merged)
- Multi-framework support documented (PyTorch, JAX, TensorFlow)
- API reference expanded
- User guide updated with extended examples
- Tutorial notebook imports updated

**New Files:**

- README.md
- api_reference.md - Complete API reference for all frameworks
- multi_framework_guide.md - Framework-specific implementation details
- data_generation_guide.md - Updated user guide with examples
- first_order_tutorial.ipynb - Updated interactive tutorial
- simple_usage.py - Updated example script
