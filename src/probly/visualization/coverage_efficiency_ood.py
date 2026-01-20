import matplotlib.pyplot as plt

from probly.visualization import CoverageEfficiencyVisualizer


def plot_coverage_efficiency_id_ood(
    probs_id,
    targets_id,
    probs_ood,
    targets_ood,
):
    viz = CoverageEfficiencyVisualizer()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    viz.plot_coverage_efficiency(
        probs_id,
        targets_id,
        title="Coverage vs Efficiency (ID)",
        ax=axes[0],
    )

    viz.plot_coverage_efficiency(
        probs_ood,
        targets_ood,
        title="Coverage vs Efficiency (OOD)",
        ax=axes[1],
    )

    plt.tight_layout()
    return axes