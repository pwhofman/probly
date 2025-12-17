# probly/conformal_prediction/framework/flax.py
# oder: probly/conformal_prediction/backends/flax.py
# -------------------------------------------------------------------
# AUS: lac/flax.py::FlaxModelWrapper (apply(params, x)) [file:11]
# AUS: aps/flax.py::FlaxModelWrapper (nnx.Module, direkt callbar), jit_predict [file:13]
# NEU: zwei Wrapper-Varianten für Flax-Modelle

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


class FlaxApplyWrapper:
    """Wrapper für klassische Flax-Module mit .apply(params, x).

    Verantwortlichkeiten:
    - Eingaben nach jax.Array/jnp.ndarray konvertieren.
    - flax_model.apply(params, x) aufrufen.
    - Softmax anwenden, um Wahrscheinlichkeiten zu erhalten.
    - Ausgabe als numpy-Array (n_samples, n_classes) zurückgeben.

    AUS: lac/flax.py::FlaxModelWrapper [file:11]
    """

    def __init__(self, flax_model: Any, params: dict[str, Any]) -> None:
        """Initialisiere Wrapper mit Flax-Modul und Parametern.

        Args:
            flax_model: Flax nn.Module mit .apply(params, x).
            params: Trainierte Parameter des Modells.
        """
        self.flax_model = flax_model
        self.params = params

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Berechne Wahrscheinlichkeiten als numpy-Array.

        Entspricht der predict-Logik aus eurem LAC-Flax-Wrapper. [file:11]
        """
        x_jax = jnp.asarray(x)
        logits = self.flax_model.apply(self.params, x_jax)
        probas_jax = jax.nn.softmax(logits, axis=-1)
        return np.asarray(probas_jax, dtype=np.float32)


class FlaxNNXWrapper:
    """Wrapper für Flax nnx.Module bzw. callbare Flax-Modelle.

    AUS: aps/flax.py (innere FlaxModelWrapper-Klasse) [file:13]
    """

    def __init__(self, flax_model: Any) -> None:
        """Initialisiere Wrapper mit einem nnx.Module oder callbarem Modell.

        Args:
            flax_model: nnx.Module oder Callable[[jax.Array], jax.Array].
        """
        self.flax_model = flax_model

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Berechne Wahrscheinlichkeiten als numpy-Array.

        Entspricht der predict-Logik aus eurer FlaxAPS-Wrapper-Klasse. [file:13]
        """
        x_array = jnp.asarray(x, dtype=jnp.float32)
        model_callable = cast("Callable[[jax.Array], jax.Array]", self.flax_model)
        logits = model_callable(x_array)
        probs = jax.nn.softmax(logits, axis=-1)
        return np.asarray(probs, dtype=np.float32)

    def jit_predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """JIT-kompilierte Vorhersage (optional, für Performance).

        AUS: FlaxAPS.jit_predict in aps/flax.py [file:13]
        """

        @jax.jit
        def predict_fn(x_input: jnp.ndarray) -> jnp.ndarray:
            logits = self.flax_model(x_input)
            return jax.nn.softmax(logits, axis=-1)

        return cast("jnp.ndarray", predict_fn(x))
