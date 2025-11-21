from __future__ import annotations

# gemeinsame Ensemble-Funktionen importieren und unter flax re-exportieren
from probly.transformation.ensemble import common

ensemble = common.ensemble
register = common.register

# Optional: falls ihr (wie bei torch) eine verzögerte Registrierung nutzt,
# kannst du das später ergänzen. Für den Test reicht der Re-Export.
# from probly.lazy_types import FLAX_MODULE
# @common.ensemble_generator.delayed_register(FLAX_MODULE)
# def _(_: type) -> None:
#     pass
