"""Configuration file for the Sphinx documentation builder."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import importlib
import inspect
import os
import sys

import probly

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../examples"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "probly"
copyright = "2025, probly team"  # noqa: A001
author = "probly team"
release = probly.__version__
version = probly.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # generates API documentation from docstrings
    "sphinx.ext.autosummary",  # generates .rst files for each module
    "sphinx_autodoc_typehints",  # optional, nice for type hints in docs
    # "sphinx.ext.linkcode",  # adds [source] links to code that link to GitHub. Use when repo is public.  # noqa: E501, ERA001
    "sphinx.ext.viewcode",  # adds [source] links to code that link to the source code in the docs.
    "sphinx.ext.napoleon",  # for Google-style docstrings
    "sphinx.ext.duration",  # optional, show the duration of the build
    "myst_nb",  # for jupyter notebook support, also includes myst_parser
    "sphinx.ext.intersphinx",  # for linking to other projects' docs
    "sphinx.ext.mathjax",  # for math support
    "sphinx.ext.doctest",  # for testing code snippets in the docs
    "sphinx_copybutton",  # adds a copy button to code blocks
    "sphinx.ext.autosectionlabel",  # for auto-generating section labels,
    "sphinxcontrib.bibtex",  # for bibliography support
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
nb_execution_mode = "off"  # don't run notebooks when building the docs

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve the link to the source code in GitHub.

    This function is required by sphinx.ext.linkcode and is used to generate links to the source code on GitHub.

    Args:
        domain (str): The domain of the object.
        info (dict[str, str]): The information about the object.

    Returns:
        str | None: The URL to the source code or None if not found.
    """
    if domain != "py" or not info["module"]:
        return None

    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        relpath = os.path.relpath(fn, start=root)
    except (ModuleNotFoundError, AttributeError, TypeError, OSError):
        return None

    base = "https://github.com/pwhofman/probly"
    tag = "v0.2.0-pre-alpha" if version == "0.2.0" else f"v{version}"

    return f"{base}/blob/{tag}/{relpath}#L{lineno}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
# TODO(pwhofman): add favicon Issue: https://github.com/pwhofman/probly/issues/95
# html_favicon = "_static/logo/"  # noqa: ERA001
pygments_dark_style = "monokai"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo/logo_light.png",
    "dark_logo": "logo/logo_dark.png",
}

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/footer.html",  # to get the github link in the footer of the sidebar
    ],
}

html_show_sourcelink = False  # to remove button next to dark mode showing source in txt format

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
# TODO(pwhofman): maybe set this to True, Issue https://github.com/pwhofman/probly/issues/94
autodoc_inherit_docstrings = False

autodoc_typehints = "both"  # to show type hints in the docstring

# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
