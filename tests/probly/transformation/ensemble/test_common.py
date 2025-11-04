from __future__ import annotations

import importlib
import pytest


def test_raises_if_no_impl_registered() -> None:
    common = importlib.import_module("probly.transformation.ensemble.common")
    importlib.reload(common)  # Registry sauber
    class Dummy: pass
    with pytest.raises(NotImplementedError, match="No ensemble generator is registered"):
        common.ensemble(Dummy(), num_members=2, reset_params=False)


def test_register_wires_generator() -> None:
    common = importlib.import_module("probly.transformation.ensemble.common")
    importlib.reload(common)
    #Dummy Generator(fake)
    calls = []
    class Dummy: pass
    def gen(obj: Dummy, *, n_members: int, reset_params: bool):
        calls.append((obj, n_members, reset_params))
        return "OK"

    common.register(Dummy, gen)
    base = Dummy()
    out = common.ensemble(base, num_members=5, reset_params=True)

    assert out == "OK"
    assert calls == [(base, 5, True)]
