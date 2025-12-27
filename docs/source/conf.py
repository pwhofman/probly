"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import probly

# -- Path setup --------------------------------------------------------------
# docs/conf.py -> Projektwurzel: ../../
DOCS_DIR = Path(__file__).resolve().parent
ROOT_DIR = DOCS_DIR.parent.parent

SRC_DIR = ROOT_DIR / "src"
EXAMPLES_DIR = ROOT_DIR / "examples"
CC_EXAMPLES_DIR = ROOT_DIR / "cc_examples"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))
if CC_EXAMPLES_DIR.exists():
    sys.path.insert(0, str(CC_EXAMPLES_DIR))

# -- Project information -----------------------------------------------------
project = "probly"
copyright = "2025, probly team"  # noqa: A001
author = "probly team"
release = probly.__version__
version = probly.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.linkcode",  # optional
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
nb_execution_mode = "off"

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
    if domain != "py" or not info.get("module"):
        return None

    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        _source, lineno = inspect.getsourcelines(obj)
        if not fn:
            return None
        relpath = os.path.relpath(fn, start=str(ROOT_DIR))
    except (ModuleNotFoundError, AttributeError, TypeError, OSError):
        return None

    base = "https://github.com/pwhofman/probly"
    tag = "v0.2.0-pre-alpha" if version == "0.2.0" else f"v{version}"
    return f"{base}/blob/{tag}/{relpath}#L{lineno}"


# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
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

html_show_sourcelink = False


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
autodoc_typehints = "both"


# -- Copy Paste Button -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True


# -- Sphinx Gallery ----------------------------------------------------------
from sphinx_gallery.sorting import FileNameSortKey  # noqa: E402

# Beispiele sammeln (cc_examples optional)
_examples_dirs: list[str] = [str(EXAMPLES_DIR)]
_gallery_dirs: list[str] = ["auto_examples"]

if CC_EXAMPLES_DIR.exists():
    _examples_dirs.append(str(CC_EXAMPLES_DIR))
    _gallery_dirs.append("auto_cc_examples")

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py",
    "within_subsection_order": FileNameSortKey,
    "download_all_examples": False,
    "default_gallery_intro": """
    "abort_on_example_error": True,
.. _example_gallery:

Example Gallery
===============

This section contains runnable examples generated with **Sphinx-Gallery**.
""",
}

