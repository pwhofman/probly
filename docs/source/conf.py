"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib
import inspect
import os
import sys

import probly

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../examples"))
sys.path.insert(0, os.path.abspath("../../cc_examples"))  # optional, aber hilfreich

# -- Project information -----------------------------------------------------
project = "probly"
copyright = "2025, probly team"  # noqa: A001
author = "probly team"
release = probly.__version__
version = probly.__version__

# -- General configuration ---------------------------------------------------
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
    "sphinx.ext.autosectionlabel",  # for auto-generating section labels
    "sphinxcontrib.bibtex",  # for bibliography support

    # ✅ Sphinx-Gallery (generiert auto_examples & auto_cc_examples)
    "sphinx_gallery.gen_gallery",
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
    """Resolve the link to the source code in GitHub."""
    if domain != "py" or not info["module"]:
        return None

    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        _source, lineno = inspect.getsourcelines(obj)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        relpath = os.path.relpath(fn, start=root)
    except (ModuleNotFoundError, AttributeError, TypeError, OSError):
        return None

    base = "https://github.com/pwhofman/probly"
    tag = "v0.2.0-pre-alpha" if version == "0.2.0" else f"v{version}"
    return f"{base}/blob/{tag}/{relpath}#L{lineno}"


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
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
        "sidebar/footer.html",
    ],
}

html_show_sourcelink = False  # remove source button in html


# -- Autodoc -----------------------------------------------------------------
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
autodoc_inherit_docstrings = False
autodoc_typehints = "both"  # show type hints in docstring


# -- Copy Paste Button -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


# -- Sphinx Gallery ----------------------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    # Quellen (relativ zu docs/)
    "examples_dirs": [
        "../examples",
        "../cc_examples",
    ],
    # Ausgabeordner (relativ zu docs/)
    "gallery_dirs": [
        "auto_examples",
        "auto_cc_examples",
    ],
    # nützliche Defaults
    "filename_pattern": r".*\.py",
    "within_subsection_order": FileNameSortKey,
    "download_all_examples": False,
}

