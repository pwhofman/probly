"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import warnings

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))  # make _sphinx_helpers importable
sys.path.insert(0, str(_here.parent / "src"))
sys.path.insert(0, str(_here.parent.parent))  # make examples.utils importable

from _sphinx_helpers import (  # noqa: E402
    add_member_source_dependencies,
    ignore_installed_template_mtimes,
    make_linkcode_resolve,
    scrub_external_dependencies,
)

import probly  # noqa: E402

if TYPE_CHECKING:
    from docutils.nodes import Element
    from sphinx.addnodes import pending_xref
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.domains.python import PythonDomain
    from sphinx.environment import BuildEnvironment

# -- Paths -------------------------------------------------------------------
# conf.py lives in:  .../probly/docs/source/conf.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../probly
print(f"REPO_ROOT: {REPO_ROOT}")  # noqa: T201

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
    "sphinx.ext.napoleon",  # for Google-style docstrings
    "sphinx.ext.duration",  # show the duration of the build
    "sphinx_gallery.gen_gallery",  # for rendering example galleries from Python scripts
    "sphinx.ext.intersphinx",  # for linking to other projects' docs
    "sphinx.ext.mathjax",  # for math support
    "sphinx.ext.doctest",  # for testing code snippets in the docs
    "sphinx_copybutton",  # adds a copy button to code blocks
    "sphinx.ext.linkcode",  # adds [source] links to code that link to GitHub.
    "sphinx.ext.autosectionlabel",  # for auto-generating section labels
    "sphinxcontrib.bibtex",  # for bibliography support
]

suppress_warnings = []

# --- Autosummary settings ----------------------------------------------------s
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# --- Autodoc settings --------------------------------------------------------
autoclass_content = "both"  # class docstring AND __init__ docstring
autodoc_typehints = "signature"  # show type hints only in the signature,
# Only show types for parameters that are actually documented
autodoc_typehints_description_target = "documented_params"  # only params that are actually documented
napoleon_use_ivar = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]

# Prefix labels with doc name: "api:Overview" instead of just "Overview"
autosectionlabel_prefix_document = True

# -- Exclude patterns ------------------------------------------------------------------------------
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/*.py",
    "**/*.json",
    "**/*.zip",
    "**/*.md5",
    # Exclude top-level notebooks directory
    "../../notebooks",
    "../../notebooks/**",
    "../../supporting_files/**",
    "sg_execution_times.rst",
]


# -- Bibtex settings -------------------------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

# -- Sphinx-Gallery --------------------------------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": [str(REPO_ROOT / "examples")],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("probly",),
    "reference_url": {"probly": None},
    "filename_pattern": r"plot_.*\.py",
    "ignore_pattern": r"(__init__|.*/llm/.*)\.py",
    "plot_gallery": True,
    "download_all_examples": False,
    "notebook_extensions": set(),
    # By default only examples whose source changed (MD5 mismatch) re-execute;
    # set FORCE_CLEAN=1 to force re-running all examples.
    "run_stale_examples": os.environ.get("FORCE_CLEAN") == "1",
    # Don't kill the whole build if one example errors
    "abort_on_example_error": False,
    # Execute examples in isolated worker processes (joblib/loky), one per
    # available CPU; set SPHINX_GALLERY_PARALLEL to override (1 disables,
    # sphinx-gallery treats it as off).
    "parallel": int(os.environ.get("SPHINX_GALLERY_PARALLEL", os.process_cpu_count() or 1)),
    "default_thumb_file": str(REPO_ROOT / "docs" / "source" / "_static" / "logo" / "logo_light.png"),
}

# loky recycles a worker by design when its memory use grows more than
# ~300MB beyond its post-first-job baseline (heavy torch examples trigger
# this); the lost job is re-run in a fresh worker, but the executor still
# warns in the main process. Harmless for the gallery build, so silence it.
warnings.filterwarnings(
    "ignore",
    message="A worker stopped while some jobs were given to the executor",
    category=UserWarning,
)

# -- Intersphinx -----------------------------------------------------------------------------------
intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/2.10/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Workaround for https://github.com/sphinx-doc/sphinx/issues/10568  --------------------------------
#
# Several bare names in signatures cause "more than one target found for
# cross-reference" warnings and wrong hyperlinks:
#
#   * PEP 695 type parameters (T, S, D, In, Out, …) — Sphinx resolves them
#     to unrelated ``.T`` properties (numpy transpose convention) instead of
#     treating them as type variables.
#   * Common attribute names (``type``) — many classes define these,
#     so Sphinx picks an arbitrary target.
#
# The ``missing-reference`` event cannot help because Sphinx *does* resolve
# the reference (ambiguously); the event only fires for truly unresolved refs.
#
# We monkey-patch ``PythonDomain.resolve_xref`` to short-circuit for these
# names, returning plain unlinked text before the domain ever searches.
_SKIP_XREF_NAMES = frozenset(
    {
        # PEP 695 usual type parameters
        "T",
        "S",
        "C",
        "D",
        "F",
        "V",
        "Q",
        "In",
        "Out",
        # Common attribute names with many targets
        "type",
        "shape",
    }
)


def setup(app: Sphinx) -> None:
    """Patch the Python domain resolver and register incremental-build hooks."""
    from sphinx.domains.python import PythonDomain  # noqa: PLC0415

    # Incremental-build accuracy hooks; see _sphinx_helpers.py docstrings.
    app.connect("env-updated", scrub_external_dependencies)
    app.connect("env-updated", add_member_source_dependencies)
    if os.environ.get("FORCE_CLEAN") != "1":
        app.connect("builder-inited", ignore_installed_template_mtimes)

    _orig_resolve = PythonDomain.resolve_xref

    def _patched_resolve(
        self: PythonDomain,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        xref_type: str,
        target: str,
        node: pending_xref,
        contnode: Element,
    ) -> Element | None:
        if target in _SKIP_XREF_NAMES:
            return contnode  # plain text, no link, no warning
        return _orig_resolve(self, env, fromdocname, builder, xref_type, target, node, contnode)

    PythonDomain.resolve_xref = _patched_resolve


linkcode_resolve = make_linkcode_resolve(REPO_ROOT)

# -- HTML output -----------------------------------------------------------------------------------
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

# -- Copy Paste Button -----------------------------------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
