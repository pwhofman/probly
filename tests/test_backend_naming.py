"""Tests for the standalone backend naming and import checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_backend_naming.py"
_spec = importlib.util.spec_from_file_location("check_backend_naming", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


@pytest.mark.parametrize("backend", ["torch", "jax", "flax", "numpy", "Torch", "Jax", "Flax", "Numpy"])
def test_backend_prefixes_and_suffixes(backend: str) -> None:
    assert checker.misplaced_backend(f"{backend}_predict") is None
    assert checker.misplaced_backend(f"_{backend}_predict") is None
    assert checker.misplaced_backend(f"predict_{backend}") == backend
    assert checker.misplaced_backend(f"_predict_{backend}_value") == backend


@pytest.mark.parametrize(
    "name",
    [
        "predict",
        "_",
        "JaxSample",
        "_FlaxPredictor",
        "numpyArray",
        "TorchHTTPModel",
        "torch_numpy_predict",
        "_jax_to_torch",
        "TorchNumpyAdapter",
        "_NumpyJaxTorchFlaxAdapter",
        "torch_torch",
        "fromNumpySample",
        "toTorchLike",
        "isJaxCompatible",
        "hasNumpyValue",
        "supportsTorch",
        "use_flaxify",
        "convert_pytorch",
        "MyFlaxifyModel",
        "PytorchModel",
        "mytorchModel",
        "foo_torched",
        "handle_numpy2",
        "SomethingJax2",
        "JAXModel",
        "MY_TORCH_MODEL",
    ],
)
def test_word_boundaries_accept_prefixes_and_substrings(name: str) -> None:
    assert checker.misplaced_backend(name) is None


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        ("predict_torch_numpy", "torch"),
        ("MyTorchModel", "Torch"),
        ("load_torchModel", "torch"),
        ("_BoundNumpyFunction", "Numpy"),
        ("PyTorchModel", "Torch"),
        ("myJax", "Jax"),
        ("HTTPFlaxAdapter", "Flax"),
        ("Model2TorchLayer", "Torch"),
        ("uses__numpy__arrays", "numpy"),
        ("__torch_predict", "torch"),
        ("_flaxify_torch", "torch"),
        ("pytorch_numpy_predict", "numpy"),
        ("SupportingTorch", "Torch"),
        ("SupportsomethingTorch", "Torch"),
        ("fromageNumpy", "Numpy"),
        ("island_torch", "torch"),
        ("this_is_torch", "torch"),
        ("try_jax_function", "jax"),
        ("handle_jax_function", "jax"),
    ],
)
def test_word_boundaries_reject_misplaced_backend_words(name: str, backend: str) -> None:
    assert checker.misplaced_backend(name) == backend


@pytest.mark.parametrize("prefix", ["from", "to", "is", "has", "supports", "Supports"])
@pytest.mark.parametrize("separator", ["", "_"])
@pytest.mark.parametrize("private", ["", "_"])
def test_semantic_prefixes(prefix: str, separator: str, private: str) -> None:
    backend = "Jax" if not separator else "jax"
    assert checker.misplaced_backend(f"{private}{prefix}{separator}{backend}_numpy_function") is None


def test_definitions_at_all_nesting_levels() -> None:
    source = """class MyTorchModel:
    def predict_numpy(self):
        async def predict_jax():
            pass
        return predict_jax
try:
    def _compute_accuracy_torch():
        pass
except ImportError:
    pass
type MyFlaxAlias[T] = list[T]
"""
    assert checker.check_source(source, "example.py") == [
        "example.py:1:1: BKN001 Backend word 'Torch' requires an allowed prefix in 'MyTorchModel'",
        "example.py:2:5: BKN001 Backend word 'numpy' requires an allowed prefix in 'predict_numpy'",
        "example.py:3:9: BKN001 Backend word 'jax' requires an allowed prefix in 'predict_jax'",
        "example.py:7:5: BKN001 Backend word 'torch' requires an allowed prefix in '_compute_accuracy_torch'",
        "example.py:11:1: BKN001 Backend word 'Flax' requires an allowed prefix in 'MyFlaxAlias'",
    ]


def test_only_definitions_are_checked_without_executing_code() -> None:
    source = '''import nonexistent_package as use_torch
from nonexistent_package import SupportsTorch as MyTorchAlias
raise RuntimeError("This source must never be executed")
my_numpy_value = "def bad_torch(): pass"
def numpy_predict(input_torch):
    """class MyTorchModel: pass"""
    # def also_bad_torch(): pass
    my_jax_value = input_torch
    return my_jax_value
'''
    assert checker.check_source(source, "example.py") == []


def test_dunder_protocol_methods_and_conversion_methods_are_exempt() -> None:
    source = """class Model:
    def __torch_function__(self): pass
    async def __jax_function__(self): pass
    def from_numpy_sample(self): pass
    def _to_torch_like(self): pass
    def is_jax_compatible(self): pass
    def has_numpy_protected_value(self): pass
    def supports_flax(self): pass
class SupportsJaxFunction: pass
type TorchNumpyAlias = int
"""
    assert checker.check_source(source, "example.py") == []


@pytest.mark.parametrize(
    "comment",
    [
        "# noqa: BKN001",
        "# noqa: ANN401, BKN001 - Protocol helper.",
        "# noqa: BKN001, ANN401",
        "# noqa: ANN401  # noqa: BKN001",
    ],
)
def test_explicit_suppression_on_opening_line(comment: str) -> None:
    source = f"""def handle_jax_function(  {comment}
    arg,
):
    pass
class MyTorchModel: pass  {comment}
type MyFlaxAlias = int  {comment}
"""
    assert checker.check_source(source, "example.py") == []


@pytest.mark.parametrize(
    "comment", ["# noqa", "# noqa: ANN401", "# noqa: BKN0010", "# noqa: BKN001extra", "# noqa: BKN002"]
)
def test_other_suppressions_do_not_hide_violations(comment: str) -> None:
    assert len(checker.check_source(f"def predict_torch(): pass  {comment}", "example.py")) == 1


def test_suppression_does_not_leak_from_strings_decorators_or_body() -> None:
    source = '''"""# noqa: BKN001"""
@decorate  # noqa: BKN001
def handle_jax_function():
    pass  # noqa: BKN001
def predict_torch(): return "# noqa: BKN001"
'''
    diagnostics = checker.check_source(source, "example.py")
    assert len(diagnostics) == 2
    assert "handle_jax_function" in diagnostics[0]
    assert "predict_torch" in diagnostics[1]


def test_suppression_applies_only_to_its_definition() -> None:
    source = """def handle_jax_function():  # noqa: BKN001 - Protocol helper.
    def bad_torch(): pass
"""
    diagnostics = checker.check_source(source, "example.py")
    assert len(diagnostics) == 1
    assert "bad_torch" in diagnostics[0]


def test_dunder_exemption_does_not_apply_to_classes_or_aliases() -> None:
    source = """class __NumpyHolder__: pass
type __TorchAlias__ = int
"""
    assert len(checker.check_source(source, "example.py")) == 2


def test_cli_default_scope_is_only_the_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "src" / "probly"
    library.mkdir(parents=True)
    (library / "model.py").write_text("class TorchModel: pass\n")
    (tmp_path / "src" / "benchmark.py").write_text("class MyTorchModel: pass\n")
    (tmp_path / "test_numpy.py").write_text("def test_numpy_function(): pass\n")
    monkeypatch.setattr(checker, "DEFAULT_ROOT", library)
    assert checker.main([]) == 0


def test_cli_reads_source_encoding_and_deduplicates_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "example.py"
    path.write_bytes('# coding: latin-1\n"caf\xe9"\ndef predict_torch(): pass\n'.encode("latin-1"))
    assert checker.main([str(tmp_path), str(path)]) == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err.count("BKN001") == 1
    assert f"{path}:3:1:" in result.err


def test_cli_reports_syntax_and_missing_file_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken.py"
    missing = tmp_path / "missing.py"
    broken.write_text("def broken(:\n")
    assert checker.main([str(broken), str(missing)]) == 1
    errors = capsys.readouterr().err
    assert f"{broken}:1:" in errors
    assert "SyntaxError:" in errors
    assert f"{missing}:1:1:" in errors


@pytest.mark.parametrize(("name", "returncode"), [("torch_predict", 0), ("predict_torch", 1)])
def test_standalone_entry_point(tmp_path: Path, name: str, returncode: int) -> None:
    source = tmp_path / "example.py"
    source.write_text(f"def {name}(): pass\n")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == returncode
    assert ("BKN001" in result.stderr) == bool(returncode)


@pytest.mark.parametrize(
    ("filename", "source"),
    [
        ("torch.py", "import torch"),
        ("torch_metrics.py", "from torch.nn import Module"),
        ("_torch_helpers.py", "import torch.nn.functional as F"),
        ("transformers.py", "import torch"),
        ("transformers_utils.py", "from torch import Tensor"),
        ("huggingface.py", "import torch"),
        ("peft_helpers.py", "import torch"),
        ("jax.py", "import jax.numpy as jnp"),
        ("_jax_functions.py", "from jax import Array"),
        ("flax.py", "import jax, flax"),
        ("flax_layers.py", "from flax.nnx import Module"),
        ("probly/torch/__init__.py", "import torch"),
        ("probly/_flax_helpers/__init__.py", "import flax, jax"),
    ],
)
def test_imports_in_approved_modules(filename: str, source: str) -> None:
    assert checker.check_source(source, filename) == []


@pytest.mark.parametrize(
    ("filename", "source", "backend"),
    [
        ("numpy.py", "import torch", "torch"),
        ("torch.py", "import jax", "jax"),
        ("jax.py", "import flax", "flax"),
        ("flax.py", "import torch", "torch"),
        ("transformers.py", "import flax", "flax"),
        ("huggingface.py", "import jax", "jax"),
        ("peft.py", "import flax", "flax"),
        ("my_torch.py", "from torch.nn import Module", "torch"),
        ("pytorch.py", "import torch", "torch"),
        ("torchvision.py", "import torch", "torch"),
        ("flaxify.py", "import flax", "flax"),
        ("__torch_helpers.py", "import torch", "torch"),
        ("probly/torch/shared.py", "import torch", "torch"),
        ("probly/common/__init__.py", "import torch", "torch"),
        ("jax.py", "import torch as jax", "torch"),
        ("common.py", "from flax.nnx import Module as TorchModule", "flax"),
    ],
)
def test_imports_in_unapproved_modules(filename: str, source: str, backend: str) -> None:
    diagnostics = checker.check_source(source, filename)
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith(f"{filename}:1:1: BKN002 Import of '{backend}'")


def test_import_roots_are_checked_independently_of_aliases_and_substrings() -> None:
    source = """import torch.nn as nn, jax.numpy as np, torch.utils
import nottorch as torch
from torch_tools import helper
from . import torch
from .torch import helper
from ..jax import helper
from probly.layers.torch import Layer
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 2
    assert "Import of 'jax'" in diagnostics[0]
    assert "Import of 'torch'" in diagnostics[1]


def test_runtime_imports_inside_functions_classes_and_try_blocks() -> None:
    source = """def predict():
    import torch
    async def inner():
        from jax import Array
class Model:
    try:
        import flax
    except ImportError:
        pass
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 3
    assert diagnostics[0].startswith("common.py:2:5: BKN002")
    assert diagnostics[1].startswith("common.py:4:9: BKN002")
    assert diagnostics[2].startswith("common.py:7:9: BKN002")


@pytest.mark.parametrize(
    ("imports", "guard"),
    [
        ("from typing import TYPE_CHECKING", "TYPE_CHECKING"),
        ("from typing import TYPE_CHECKING as TC", "TC"),
        ("import typing", "typing.TYPE_CHECKING"),
        ("import typing as t", "t.TYPE_CHECKING"),
        ("from typing_extensions import TYPE_CHECKING as TC", "TC"),
    ],
)
@pytest.mark.parametrize("negated", [False, True])
def test_type_only_branches_are_exempt_but_runtime_branches_are_checked(
    imports: str, guard: str, negated: bool
) -> None:
    condition = f"not {guard}" if negated else guard
    type_only = "import torch, jax, flax"
    runtime = "import torch"
    body, otherwise = (runtime, type_only) if negated else (type_only, runtime)
    source = f"{imports}\nif {condition}:\n    {body}\nelse:\n    {otherwise}\n"
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 1
    runtime_line = 3 if negated else 5
    assert diagnostics[0].startswith(f"common.py:{runtime_line}:5: BKN002")


def test_type_only_import_exemption_does_not_suppress_definition_naming() -> None:
    source = """from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    class BadTorch: pass
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 1
    assert "BKN001" in diagnostics[0]


@pytest.mark.parametrize(
    "source",
    [
        "TYPE_CHECKING = True\nif TYPE_CHECKING:\n    import torch\n",
        "import other as typing\nif typing.TYPE_CHECKING:\n    import torch\n",
        "from .typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n",
        "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\nif TYPE_CHECKING:\n    import torch\n",
        "from typing import TYPE_CHECKING\ndef f(TYPE_CHECKING):\n    if TYPE_CHECKING:\n        import torch\n",
        "from typing import TYPE_CHECKING\nimport other as TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n",
    ],
)
def test_unrecognized_or_shadowed_type_checking_guards_are_not_exempt(source: str) -> None:
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 1
    assert "BKN002" in diagnostics[0]


def test_type_checking_aliases_stay_in_their_scope() -> None:
    source = """def helper():
    from typing import TYPE_CHECKING as TC
    if TC:
        import torch
if TC:
    import jax
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("common.py:6:5: BKN002 Import of 'jax'")


def test_conditional_aliases_do_not_exempt_runtime_imports_in_other_branches() -> None:
    source = """TC = True
if flag:
    from typing import TYPE_CHECKING as TC
else:
    if TC:
        import torch
if TC:
    import jax
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 2
    assert diagnostics[0].startswith("common.py:6:9: BKN002 Import of 'torch'")
    assert diagnostics[1].startswith("common.py:8:5: BKN002 Import of 'jax'")


def test_import_suppression_uses_the_opening_line_and_only_its_rule() -> None:
    source = """from torch import (  # noqa: BKN002 - Intentional bridge.
    Tensor,
)
def bad_jax(): pass  # noqa: BKN002
import jax  # noqa: BKN001
import flax  # noqa: BKN001, BKN002
"""
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 2
    assert diagnostics[0].startswith("common.py:4:1: BKN001")
    assert diagnostics[1].startswith("common.py:5:1: BKN002 Import of 'jax'")


@pytest.mark.parametrize("comment", ["# noqa", "# noqa: BKN001", "# noqa: BKN0020", "# noqa: BKN002extra"])
def test_import_suppression_requires_the_exact_rule(comment: str) -> None:
    diagnostics = checker.check_source(f"import torch  {comment}", "common.py")
    assert len(diagnostics) == 1
    assert "BKN002" in diagnostics[0]


def test_import_strings_and_comments_do_not_count_as_imports_or_suppressions() -> None:
    source = '''"import torch"
# from jax import Array
"""# noqa: BKN002"""
import flax
'''
    diagnostics = checker.check_source(source, "common.py")
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("common.py:4:1: BKN002 Import of 'flax'")


def test_cli_reports_backend_import_violations(tmp_path: Path) -> None:
    source = tmp_path / "common.py"
    source.write_text("import torch\n")
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), str(source)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "BKN002" in result.stderr
