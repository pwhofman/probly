"""Configuration file for the Sphinx documentation builder."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "probly"
copyright = "2025, probly team"  # noqa: A001
author = "probly team"
release = "0.2.0"
version = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # generates API documentation from docstrings
    "sphinx.ext.autosummary",  # generates .rst files for each module
    "sphinx_autodoc_typehints",  # optional, nice for type hints in docs
    "sphinx.ext.viewcode",  # adds [source] links to code
    "sphinx.ext.napoleon",  # for Google-style docstrings
    "sphinx.ext.duration",  # optional, show the duration of the build
    "myst_parser",  # for markdown support
    "sphinx.ext.intersphinx",  # for linking to other projects' docs
    "sphinx.ext.mathjax",  # for math support
    "sphinx.ext.doctest",  # for testing code snippets in the docs
    "sphinx_copybutton",  # adds a copy button to code blocks
    "sphinx.ext.autosectionlabel",  # for auto-generating section labels
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
# html_favicon = "_static/logo/"  # TODO: add favicon  # noqa: ERA001
pygments_dark_style = "monokai"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo/probly_logo.svg",
    "dark_logo": "logo/probly_logo.svg",
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Autodoc ---------------------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "class"
autodoc_inherit_docstrings = False  # TODO: maybe set this to True
autodoc_typehints = "both"  # to show type hints in the docstring

# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
