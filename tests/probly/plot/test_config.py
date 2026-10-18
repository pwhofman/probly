"""Tests for ``probly.plot.config``."""

from __future__ import annotations

import subprocess
import sys

import matplotlib as mpl
import pytest

from probly.plot import PlotConfig, use_probly_style


class TestPlotConfig:
    """PlotConfig: defaults and color-cycling palette behaviour."""

    def test_color_cycles_with_modulus(self) -> None:
        cfg = PlotConfig()
        n = len(cfg.categorical_palette)
        assert cfg.color(0) == cfg.categorical_palette[0]
        assert cfg.color(n) == cfg.categorical_palette[0]
        assert cfg.color(n + 2) == cfg.categorical_palette[2]

    def test_default_palette_is_non_empty(self) -> None:
        cfg = PlotConfig()
        assert len(cfg.categorical_palette) >= 2
        # All entries should look like hex colours.
        for c in cfg.categorical_palette:
            assert isinstance(c, str)
            assert c.startswith("#")

    def test_immutable(self) -> None:
        cfg = PlotConfig()
        with pytest.raises(Exception):  # noqa: B017,PT011
            cfg.figure_size = (10.0, 10.0)  # type: ignore[misc]


class TestUseProblyStyle:
    """use_probly_style: opt-in global style instead of an import-time side effect."""

    def test_import_leaves_rcparams_unchanged(self) -> None:
        program = (
            "import matplotlib as mpl; "
            "before = dict(mpl.rcParams); "
            "import probly.plot; "
            "changed = sorted(k for k, v in mpl.rcParams.items() if before.get(k) != v); "
            "assert not changed, changed"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    def test_use_probly_style_sets_rcparams(self) -> None:
        with mpl.rc_context():
            use_probly_style()
            assert mpl.rcParams["font.sans-serif"][0] == "Fira Sans"
            assert mpl.rcParams["axes.titlesize"] == 14

    def test_repeated_calls_keep_one_fira_sans_entry(self) -> None:
        with mpl.rc_context():
            use_probly_style()
            use_probly_style()
            assert mpl.rcParams["font.sans-serif"].count("Fira Sans") == 1
