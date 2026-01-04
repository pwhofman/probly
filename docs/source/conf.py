"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import probly

# -- Paths -------------------------------------------------------------------
# conf.py lives in:  .../probly/docs/source/conf.py
DOCS_SOURCE_DIR = Path(__file__).resolve().parent          # .../docs/source
DOCS_DIR = DOCS_SOURCE_DIR.parent                         # .../docs
REPO_ROOT = DOCS_DIR.parent                               # .../probly

# Add package + example dirs to Python path
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "cc_examples"))

# -- Project information -----------------------------------------------------
project = "probly"
author = "probly team"
copyright = "2025, probly team"  # noqa: A001

release = probly.__version__
version = probly.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx_gallery.gen_gallery",
    "sphinx_gallery.load_style",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Ignore generated artifacts if they end up in the source tree
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
    "auto_examples/*.json",
    "auto_examples/*.db",
    "auto_examples/*.md5",
    "auto_examples/cc_examples/*.ipynb",
    "auto_examples/cc_examples/*.py",
    "auto_examples/cc_examples/*.zip",
    "auto_examples/cc_examples/*.json",
    "auto_examples/cc_examples/*.db",
    "auto_examples/cc_examples/*.md5",
]

# Notebooks: don't execute during docs build
nb_execution_mode = "off"

# Bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

# -- Sphinx-Gallery ----------------------------------------------------------
# Two galleries:
# - REPO_ROOT/examples     -> docs/source/auto_examples/examples
# - REPO_ROOT/cc_examples  -> docs/source/auto_examples/cc_examples
sphinx_gallery_conf = {
    "examples_dirs": [
        str(REPO_ROOT / "examples"),
        str(REPO_ROOT / "cc_examples"),
    ],
    "gallery_dirs": [
        "auto_examples/examples",
        "auto_examples/cc_examples",
    ],
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("probly",),
    "reference_url": {"probly": None},
    "filename_pattern": r"plot_.*\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    # Don’t kill the whole build if one example errors
    "abort_on_example_error": False,
    "default_thumb_file": str(DOCS_DIR / "_static" / "logo" / "logo_light.png"),
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Linkcode (optional) -----------------------------------------------------
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    if domain != "py" or not info.get("module"):
        return None
    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        _src, lineno = inspect.getsourcelines(obj)
        if fn is None:
            return None
        relpath = os.path.relpath(fn, start=str(REPO_ROOT))
    except (ModuleNotFoundError, AttributeError, TypeError, OSError):
        return None

    base = "https://github.com/n-teGruppe/probly"
    branch = "sphinx_gallery"
    return f"{base}/blob/{branch}/{relpath}#L{lineno}"

# -- HTML output -------------------------------------------------------------
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
autosummary_generate = False
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "inherited-members": True,
    "member-order": "groupwise",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autoclass_content = "class"
autodoc_inherit_docstrings = False
autodoc_typehints = "both"

# -- Copybutton --------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Linkcheck ---------------------------------------------------------------
linkcheck_ignore = [
    r"https://doi.org/10.1142/S0218488500000253",
    r"https://www.worldscientific.com/.*",
    r"https://doi.org/10.1080/03081070500473490",
    r"https://www.tandfonline.com/.*",
]
