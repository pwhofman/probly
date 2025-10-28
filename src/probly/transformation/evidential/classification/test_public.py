

import os
import sys
import importlib
import types


def _ensure_src_on_syspath():
    # <projektwurzel>/tests/... → <projektwurzel>/src
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "../../../../..", "src"))
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_syspath()


def _import_pkg():
    return importlib.import_module("probly.transformation.evidential.classification")


def test_modul_importierbar():
    pkg = _import_pkg()
    assert isinstance(pkg, types.ModuleType)


def test_oeffentliche_api_vorhanden():
    pkg = _import_pkg()
    assert hasattr(pkg, "evidential_classification")
    assert hasattr(pkg, "register")


def test_oeffentliche_api_aufrufbar():
    pkg = _import_pkg()
    assert callable(pkg.evidential_classification)
    assert callable(pkg.register)
