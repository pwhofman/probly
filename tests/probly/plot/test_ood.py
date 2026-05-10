"""Tests for ``probly.plot.ood``: histogram, ROC, PR, and ``_resolve_axes``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from probly.plot import PlotConfig, plot_histogram, plot_pr_curve, plot_roc_curve
from probly.plot.ood import _resolve_axes


@pytest.mark.usefixtures("_close_figures")
class TestOoDPlots:
    """Visualisation helpers for OOD evaluation."""

    def test_plot_histogram_creates_two_groups(self) -> None:
        rng = np.random.default_rng(0)
        id_scores = rng.standard_normal(200)
        ood_scores = rng.standard_normal(200) + 2.0
        fig = plot_histogram(id_scores, ood_scores, title="Hist")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Score"
        assert ax.get_ylabel() == "Density"
        assert ax.get_title() == "Hist"
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "In-Distribution" in legend_texts
        assert "Out-of-Distribution" in legend_texts

    def test_plot_histogram_with_explicit_axes(self) -> None:
        fig, ax = plt.subplots()
        result = plot_histogram(np.zeros(5), np.ones(5), ax=ax)
        # The returned figure should be the one we created.
        assert result is fig

    def test_plot_roc_curve(self) -> None:
        fpr = np.linspace(0, 1, 50)
        tpr = np.sqrt(fpr)
        fig = plot_roc_curve(fpr, tpr, auroc=0.85)
        ax = fig.axes[0]
        assert ax.get_title() == "ROC Curve"
        # Legend label should contain the AUROC.
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("AUROC" in t for t in legend_texts)

    def test_plot_roc_curve_with_fpr95(self) -> None:
        fpr = np.linspace(0, 1, 20)
        tpr = np.sqrt(fpr)
        fig = plot_roc_curve(fpr, tpr, auroc=0.9, fpr95=0.05)
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("FPR@95" in t for t in legend_texts)

    def test_plot_pr_curve(self) -> None:
        recall = np.linspace(0, 1, 30)
        precision = 1 - 0.5 * recall
        fig = plot_pr_curve(recall, precision, aupr=0.7)
        ax = fig.axes[0]
        assert ax.get_title() == "Precision-Recall Curve"

    def test_resolve_axes_creates_new_figure_when_ax_none(self) -> None:
        cfg = PlotConfig()
        fig, ax = _resolve_axes(None, cfg)
        assert fig is not None
        assert ax is not None
        # The returned figure should be the parent of the axes.
        assert ax.figure is fig
