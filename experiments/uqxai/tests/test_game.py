"""Tests of the superpixel game, the coalition cache and the Shapley values of vector games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapiq
import torch
from uqxai import readouts
from uqxai.explain import shapley_values
from uqxai.game import CoalitionCache, SuperpixelMasker

from probly.quantification import quantify
from probly.representation.distribution import create_categorical_distribution_from_logits
from probly.representation.sample import create_sample

if TYPE_CHECKING:
    from pathlib import Path


def test_masker_replaces_absent_superpixels() -> None:
    image = torch.ones(3, 10, 10)
    masker = SuperpixelMasker(image, patch_size=4)
    assert masker.grid_shape == (3, 3)
    assert masker.n_players == 9
    coalitions = np.zeros((2, 9), dtype=bool)
    coalitions[1, [0, 8]] = True
    masked = masker(coalitions)
    assert masked[0].abs().sum() == 0
    assert masked[1, :, :4, :4].eq(1).all()
    assert masked[1, :, 8:, 8:].eq(1).all()
    assert masked[1].sum() == 3 * (16 + 4)


def _weights(n_players: int = 8) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n_players, 3))


def test_cache_evaluates_each_coalition_once(tmp_path: Path) -> None:
    weights = _weights()
    calls: list[int] = []

    def evaluate(coalitions: np.ndarray) -> dict[str, np.ndarray]:
        calls.append(len(coalitions))
        return {"head_logits": (coalitions @ weights)[:, None, :]}

    cache = CoalitionCache(8, evaluate, batch_size=4)
    coalitions = np.random.default_rng(1).random((10, 8)) < 0.5
    first = cache(np.concatenate([coalitions, coalitions]))["head_logits"]
    assert sum(calls) == len(np.unique(coalitions, axis=0)) == len(cache)
    assert np.allclose(first[:10], first[10:])
    assert np.allclose(first[:10, 0], coalitions @ weights)
    cache(coalitions)
    assert sum(calls) == len(cache)

    cache.save(tmp_path / "cache.npz")
    restored = CoalitionCache.load(tmp_path / "cache.npz")
    assert np.allclose(restored(coalitions[::-1])["head_logits"], first[:10][::-1])


def test_shapley_values_of_probabilities_sum_to_zero_over_classes() -> None:
    weights = _weights()

    def game(coalitions: np.ndarray) -> np.ndarray:
        return readouts.probability((coalitions @ weights)[:, None, :])

    attribution = shapley_values(game, 8, budget=2**8, approximator="kernelshap")
    exact = np.column_stack(
        [shapiq.ExactComputer(lambda c, k=k: game(c)[:, k], n_players=8)("SV").get_n_order_values(1) for k in range(3)]
    )
    assert np.allclose(attribution.values, exact, atol=1e-6)
    assert np.allclose(attribution.values.sum(axis=1), 0, atol=1e-6)
    assert np.allclose(attribution.values.sum(axis=0), attribution.full - attribution.empty, atol=1e-6)


def test_uncertainty_readout_matches_probly() -> None:
    logits = torch.randn(6, 5, 3, generator=torch.Generator().manual_seed(0))
    sample = create_sample([create_categorical_distribution_from_logits(logits[:, head]) for head in range(5)])
    decomposition = quantify(sample)
    expected = torch.stack([decomposition.total, decomposition.aleatoric, decomposition.epistemic], dim=1)
    assert np.allclose(readouts.uncertainty(logits.numpy()), expected.numpy(), atol=1e-5)
