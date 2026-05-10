"""Tests for ``probly.plot.config.PlotConfig``."""

from __future__ import annotations

import pytest

from probly.plot import PlotConfig


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
