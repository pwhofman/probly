"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import probly

# -- Paths -------------------------------------------------------------------
# conf.py lives in:  .../probly/docs/source/conf.py
DOCS_SOURCE_DIR = Path(__file__).resolve().parent  # .../docs/source
DOCS_DIR = DOCS_SOURCE_DIR.parent  # .../docs
REPO_ROOT = DOCS_DIR.parent  # .../probly

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
    "sphinx.ext.autodoc",  # generates API documentation from docstrings
    "sphinx.ext.autosummary",  # generates .rst files for each module
    "sphinx.ext.viewcode",  # adds [source] links to code that link to the source code in the docs.
    "sphinx.ext.napoleon",  # for Google-style docstrings
    "sphinx.ext.duration",  # optional, show the duration of the build
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",  # for linking to other projects' docs
    "sphinx.ext.mathjax",  # for math support
    "sphinx.ext.doctest",  # for testing code snippets in the docs
    "sphinx_copybutton",  # adds a copy button to code blocks
    # "sphinx.ext.linkcode",  # adds [source] links to code that link to GitHub. Use when repo is public.  # noqa: E501, ERA001
    "myst_nb",  # for jupyter notebook support, also includes myst_parser
    # "sphinx.ext.autosectionlabel",  # for auto-generating section labels
    "sphinxcontrib.bibtex",  # for bibliography support
]

suppress_warnings = [
    "toc.not_included",
    "autodoc.import_object",
    "autodoc",
    "ref.ref",
    "ref.doc",
    "ref.python",
    "misc.highlighting_failure",
    "myst.header",
    "autosummary",
    "toc.not_readable",
    "docutils",
]

templates_path = ["_templates"]

# Notebooks: don't execute during docs build
nb_execution_mode = "off"

# Bibliography
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/*.py", "**/*.json", "**/*.zip", "**/*.md5"]
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

# -- Sphinx-Gallery ----------------------------------------------------------
# Two galleries rendered into auto_examples/ and its cc_examples subfolder.
sphinx_gallery_conf = {
    "examples_dirs": [
        str(REPO_ROOT / "examples"),
        str(REPO_ROOT / "cc_examples"),
    ],
    "gallery_dirs": [
        "auto_examples",
        "auto_examples/cc_examples",
    ],
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("probly",),
    "reference_url": {"probly": None},
    "filename_pattern": r"plot_.*\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    # Don't kill the whole build if one example errors
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
    """Return a URL to the source for the given Python object for Sphinx linkcode.

    Parameters
    ----------
    domain : str
        The domain, e.g. "py".
    info : dict[str, str]
        Info dict containing keys like "module" and "fullname".

    Returns:
    -------
    str | None
        The URL to the source file (or None if it cannot be resolved).
    """
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

autodoc_typehints = "description"  # put typehints in the description instead of the signature

# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Linkcheck ---------------------------------------------------------------
linkcheck_ignore = [
    r"https://doi.org/10.1142/S0218488500000253",
    r"https://www.worldscientific.com/.*",
    r"https://doi.org/10.1080/03081070500473490",
    r"https://www.tandfonline.com/.*",
]
