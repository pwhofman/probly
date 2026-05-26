"""Tests for the torch-backed conformal set classes."""

from __future__ import annotations

import pytest


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestTorchOneHotConformalSet:
    """One-hot conformal sets backed by torch tensors."""

    def test_from_bool_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        tensor = torch.tensor([[True, False, True], [False, True, False]])
        s = TorchOneHotConformalSet(tensor=tensor)
        torch.testing.assert_close(s.set_size, torch.tensor([2, 1]))

    def test_from_int_tensor(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        tensor = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int64)
        s = TorchOneHotConformalSet(tensor=tensor)
        # Int tensors get coerced to bool internally.
        assert s.tensor.dtype == torch.bool

    def test_invalid_tensor_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(ValueError, match="one-hot encoded"):
            TorchOneHotConformalSet(tensor=torch.tensor([[2, 1]], dtype=torch.int64))

    def test_from_tensor_sample_factory(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        s = TorchOneHotConformalSet.from_tensor_sample(torch.tensor([[True, False]]))
        assert isinstance(s, TorchOneHotConformalSet)

    def test_from_tensor_sample_with_non_tensor_raises(self) -> None:
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            TorchOneHotConformalSet.from_tensor_sample([[True, False]])  # type: ignore[arg-type]

    def test_from_sample_factory(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        sample = TorchSample(tensor=torch.tensor([[True, False]]), sample_dim=0)
        s = TorchOneHotConformalSet.from_sample(sample)
        assert isinstance(s, TorchOneHotConformalSet)

    def test_from_sample_non_torch_sample_raises(self) -> None:
        from probly.representation.conformal_set.torch import TorchOneHotConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="TorchSample"):
            TorchOneHotConformalSet.from_sample("not-a-sample")  # type: ignore[arg-type]


class TestTorchIntervalConformalSet:
    """Interval conformal sets backed by torch tensors."""

    def test_from_tensor_samples(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchIntervalConformalSet  # noqa: PLC0415

        lower = torch.tensor([1.0, 2.0])
        upper = torch.tensor([2.0, 3.0])
        s = TorchIntervalConformalSet.from_tensor_samples(lower, upper)
        torch.testing.assert_close(s.set_size, torch.tensor([1.0, 1.0]))

    def test_from_tensor_samples_non_tensor_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            TorchIntervalConformalSet.from_tensor_samples([1.0, 2.0], torch.tensor([3.0, 4.0]))  # type: ignore[arg-type]

    def test_from_samples_factory(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchIntervalConformalSet  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        lower = TorchSample(tensor=torch.tensor([1.0, 2.0]), sample_dim=0)
        upper = TorchSample(tensor=torch.tensor([2.0, 3.0]), sample_dim=0)
        s = TorchIntervalConformalSet.from_samples(lower, upper)
        torch.testing.assert_close(s.set_size, torch.tensor([1.0, 1.0]))

    def test_from_samples_non_sample_raises(self) -> None:
        torch = _torch_modules()
        from probly.representation.conformal_set.torch import TorchIntervalConformalSet  # noqa: PLC0415

        with pytest.raises(TypeError, match="TorchSample"):
            TorchIntervalConformalSet.from_samples(torch.tensor([1.0]), torch.tensor([2.0]))  # type: ignore[arg-type]
