#!/usr/bin/env python3
"""Smoke-check that selective prediction runs through on every installed backend.

Run with ``uv run python scripts/check_selective_prediction_backends.py``. Backends that are not installed are
skipped. Exits with 1 if a selector or selective predictor raises on a backend.
"""

from __future__ import annotations

from importlib.util import find_spec
import sys
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np

from probly.representer import representer  # ty: ignore[unresolved-import]
from probly.selective_prediction import selective_predictor, selector  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from collections.abc import Callable


def _numpy_backend() -> tuple[Callable[[np.ndarray], Any], Callable[[np.ndarray, np.ndarray], Any]]:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    return np.asarray, lambda x, y: LogisticRegression().fit(x, y)


def _torch_backend() -> tuple[Callable[[np.ndarray], Any], Callable[[np.ndarray, np.ndarray], Any]]:
    import torch  # noqa: PLC0415

    from probly.transformation import ensemble  # noqa: PLC0415

    return torch.from_numpy, lambda _x, _y: ensemble(
        torch.nn.Linear(4, 3), num_members=3, predictor_type="logit_classifier"
    )


def _flax_backend() -> tuple[Callable[[np.ndarray], Any], Callable[[np.ndarray, np.ndarray], Any]]:
    from flax import nnx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from probly.method import cast  # noqa: PLC0415

    return jnp.asarray, lambda _x, _y: cast(nnx.Linear(4, 3, rngs=nnx.Rngs(0)), predictor_type="logit_classifier")


BACKENDS = {"numpy": ("sklearn", _numpy_backend), "torch": ("torch", _torch_backend), "flax": ("flax", _flax_backend)}


def _check(name: str, step: str, run: Callable[[], object]) -> bool:
    try:
        run()
    except Exception:  # noqa: BLE001
        sys.stdout.write(f"{name:6} {step:28} FAILED\n")
        traceback.print_exc(limit=1)
        return False
    sys.stdout.write(f"{name:6} {step:28} OK\n")
    return True


def main() -> int:
    """Run the checks and return the exit code."""
    rng = np.random.default_rng(0)
    scores, losses = rng.random(50), (rng.random(50) < 0.1).astype(np.float64)
    x, y = rng.normal(size=(40, 4)).astype(np.float32), rng.integers(0, 3, 40)
    ok = True
    for name, (module, load) in BACKENDS.items():
        if find_spec(module) is None:
            sys.stdout.write(f"{name:6} skipped, {module} is not installed\n")
            continue
        convert, make_model = load()
        u, err = convert(scores), convert(losses)
        rules = {
            "threshold": lambda: selector(threshold=0.5),
            "coverage": lambda u=u: selector(coverage=0.8).calibrate(u),
            "risk": lambda u=u, err=err: selector(risk=0.3, delta=0.1).calibrate(u, err),
        }
        for rule, make in rules.items():
            ok &= _check(name, f"selector {rule}", lambda make=make, u=u: make().select(u))

        def predictor(convert: Callable[[np.ndarray], Any] = convert, make_model: Any = make_model) -> None:  # noqa: ANN401
            sp = selective_predictor(representer(make_model(x, y)), selector=selector(coverage=0.8)).calibrate(
                None, convert(x)
            )
            sp.predict(convert(x))
            sp.evaluate(convert(y), convert(x))

        ok &= _check(name, "selective_predictor", predictor)
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
