"""Demo for Coverage-Efficiency plot using OOD-style labels (0=ID, 1=OOD)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.visualization.efficiencyviz.coverage_efficiency_ood import plot_coverage_efficiency_from_ood_labels


def main() -> None:
    """Run a small demo for the Coverage-Efficiency ID/OOD bridge plot."""
    n_id, n_ood, c = 400, 300, 10
    rng = np.random.default_rng(0)

    # ID: sharper (more confident) probabilities
    probs_id = rng.dirichlet(np.ones(c) * 3.0, size=n_id)
    targets_id = rng.integers(0, c, size=n_id)

    # OOD: flatter (less confident) probabilities
    probs_ood = rng.dirichlet(np.ones(c) * 1.0, size=n_ood)
    targets_ood = rng.integers(0, c, size=n_ood)

    probs = np.concatenate([probs_id, probs_ood], axis=0)
    targets = np.concatenate([targets_id, targets_ood], axis=0)

    # 0=ID, 1=OOD (same convention as OOD API)
    ood_labels = np.concatenate(
        [np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)],
        axis=0,
    )

    plot_coverage_efficiency_from_ood_labels(probs, targets, ood_labels)
    plt.show()


if __name__ == "__main__":
    main()
