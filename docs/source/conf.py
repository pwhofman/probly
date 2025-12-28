"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib
import inspect
import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../examples"))
sys.path.insert(0, os.path.abspath("../../cc_examples"))

import probly

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
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "myst_nb",
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
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
    "auto_examples/*.json",
    "auto_examples/*.db",
    "auto_examples/*.md5",
    "auto_cc_examples/*.ipynb",
    "auto_cc_examples/*.py",
    "auto_cc_examples/*.zip",
    "auto_cc_examples/*.json",
    "auto_cc_examples/*.db",
    "auto_cc_examples/*.md5",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
nb_execution_mode = "off"

# Keep example scripts at repo root so they can be used outside docs as well.
sphinx_gallery_conf = {
<<<<<<< Updated upstream
    # Keep example scripts at repo root so they can be used outside docs as well.
=======
>>>>>>> Stashed changes
    "examples_dirs": [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cc_examples")),
    ],
<<<<<<< Updated upstream

    # Sphinx-Gallery writes generated .rst and thumbnails here (relative to this conf.py).
    "gallery_dirs": "auto_examples",

    # Enables backreference pages, which power the `.. minigallery:: some.object` directive.
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("probly",),
    "reference_url": {"probly": No_

    # Sphinx-Gallery writes generated .rst and thumbnails here (relative to this conf.py).
   "gallery_dirs": "auto_examples",

    # Enables backreference pages, which power the `.. minigallery:: some.object` directive.
    "backreferences_dir": "generated/backreferences",
    # Tell Sphinx-Gallery which modules are "yours" for cross-referencing.
=======
    "gallery_dirs": [
        "auto_examples",
        "auto_cc_examples",
    ],

    # backreferences are required for .. minigallery::
    # Use separate backrefs per gallery to avoid collisions.
    "backreferences_dir": 
        "generated/backreferences_examples",

>>>>>>> Stashed changes
    "doc_module": ("probly",),
    "reference_url": {"probly": None},

    "minigallery_sort_order": lambda filename: (
        {
            "plot_gallery_smoke_test.py": 0,
            "plot_create_sample_dispatch.py": 1,
            "plot_using_predict_protocol.py": 2,
            "plot_samples_with_array_sample.py": 3,
        }.get(os.path.basename(str(filename)), 10_000),
        os.path.basename(str(filename)),
    ),
    "within_subsection_order": lambda filename: (
        {
            "plot_gallery_smoke_test.py": 0,
            "plot_create_sample_dispatch.py": 1,
            "plot_using_predict_protocol.py": 2,
            "plot_samples_with_array_sample.py": 3,
        }.get(os.path.basename(str(filename)), 10_000),
        os.path.basename(str(filename)),
    ),

    "filename_pattern": r"plot_",
    "plot_gallery": True,
    "default_thumb_file": os.path.join(
        os.path.dirname(__file__), "_static", "logo", "logo_light.png"
    ),
    "download_all_examples": False,
}

intersphinx_mapping = {
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
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
html_theme_options = {
    "show_toc_level": 2,
}
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

# -- Autodoc ----------------------------------------------------------------
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

# -- Copy Paste Button -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

linkcheck_ignore = [
    r"https://doi.org/10.1142/S0218488500000253",
    r"https://www.worldscientific.com/.*",
    r"https://doi.org/10.1080/03081070500473490",
    r"https://www.tandfonline.com/.*",
]