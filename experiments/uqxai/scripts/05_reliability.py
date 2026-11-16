"""Measure how reliable the attributions are, from two independent coalition samples.

Needs ``03_explain.py`` runs with ``--seed 0`` and ``--seed 1`` for the configuration. Everything
is recomputed from the stored coalitions, without the model. Two diagnostics:

- Split-half reliability: the correlation of the attributions from the two seeds.
- The common part of the attributions of the class probabilities. The probabilities sum to one for
  every coalition, so exact Shapley values sum to zero over the classes; any common part is error.

KernelSHAP is linear in the game's values and so keeps the sum at zero when all classes share their
coalitions. OddSHAP selects its interaction terms per class and does not; ``oddshap_projected``
removes the common part afterwards.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import shapiq
from uqxai import simplex
from uqxai.explain import shapley_values
from uqxai.game import CoalitionCache
from uqxai.paths import RESULT_DIR
from uqxai.readouts import READOUTS


def main() -> None:
    """Compute the diagnostics of both approximators and save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="result directory below results/, e.g. electric/dog_guitar_patch16_mean")
    parser.add_argument("--readout", default="probability")
    args = parser.parse_args()

    run_dir = RESULT_DIR / args.run
    budget = json.loads((run_dir / "summary.json").read_text())["budget"]
    cache = CoalitionCache.load(run_dir / "coalitions.npz")
    readout = READOUTS[args.readout]

    def game(coalitions: np.ndarray) -> np.ndarray:
        return readout(cache(coalitions)["head_logits"])

    # KernelSHAP with OddSHAP's sampling reads the same stored coalitions.
    odd_sampling = {
        "sampling_weights": shapiq.OddSHAP._init_sampling_weights_static(cache.n_players),  # noqa: SLF001
        "pairing_trick": True,
    }
    estimates = {
        "oddshap": [shapley_values(game, cache.n_players, budget, random_state=seed).values for seed in (0, 1)],
        "kernelshap": [
            shapley_values(game, cache.n_players, budget, "kernelshap", random_state=seed, **odd_sampling).values
            for seed in (0, 1)
        ],
    }
    estimates["oddshap_projected"] = [simplex.contrast(values) for values in estimates["oddshap"]]

    report = {}
    for name, (first, second) in estimates.items():
        report[name] = {
            "split_half_correlation": float(np.corrcoef(first.ravel(), second.ravel())[0, 1]),
            "split_half_rmse_relative_to_largest": float(np.sqrt(np.mean((first - second) ** 2)) / np.abs(first).max()),
            "share_of_squared_magnitude_that_is_common": float(
                (simplex.common_magnitude(first) ** 2).sum() / (simplex.magnitude(first) ** 2).sum()
            ),
        }
    report["correlation_oddshap_kernelshap_seed0"] = float(
        np.corrcoef(estimates["oddshap"][0].ravel(), estimates["kernelshap"][0].ravel())[0, 1]
    )
    (run_dir / f"reliability_{args.readout}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
