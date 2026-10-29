"""Tests for the torch backend of ``calculate_quantile``."""

from __future__ import annotations

import pytest


def _torch():
    return pytest.importorskip("torch")


class TestQuantileTorch:
    """`calculate_quantile` for torch tensors."""

    def test_torch_quantile_runs(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
        q = calculate_quantile(scores, alpha=0.1)
        assert isinstance(q, float)

    def test_torch_alpha_out_of_range_raises(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(torch.tensor([0.1, 0.2], dtype=torch.float32), alpha=1.5)

    def test_torch_empty_scores_raises(self) -> None:
        torch = _torch()
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="empty"):
            calculate_quantile(torch.tensor([], dtype=torch.float32), alpha=0.1)

    def test_torch_weighted_quantile_unweighted(self) -> None:
        torch = _torch()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calculate_weighted_quantile(values, 0.5) == pytest.approx(3.0)

    def test_torch_weighted_quantile_with_weights(self) -> None:
        torch = _torch()
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([1.0, 0.0, 0.0])
        result = calculate_weighted_quantile(values, 0.5, sample_weight=weights)
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("dtype_name", ["float32", "float64"])
    def test_torch_quantile_matches_numpy_for_any_dtype(self, dtype_name: str) -> None:
        torch = _torch()
        import numpy as np  # noqa: PLC0415

        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        values = [0.31, 0.05, 0.77, 0.12, 0.58, 0.9, 0.44]
        scores = torch.tensor(values, dtype=getattr(torch, dtype_name))

        assert calculate_quantile(scores, alpha=0.2) == pytest.approx(calculate_quantile(np.array(values), alpha=0.2))

    def test_torch_quantile_on_cuda_scores(self) -> None:
        torch = _torch()
        if not torch.cuda.is_available():
            pytest.skip("CUDA device required")
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device="cuda")

        assert calculate_quantile(scores, alpha=0.1) == calculate_quantile(scores.cpu(), alpha=0.1)
