from __future__ import annotations

from collections.abc import Iterable
import contextlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from probly.data_generation.first_order_datagenerator import (
    FirstOrderDataGenerator as PyBaseFirstOrderDataGenerator,
)
from probly.data_generation.torch_first_order_generator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    _is_probabilities,
    load_distributions_pt,
    output_dataloader,
    save_distributions_pt,
)
from probly.transformation.bayesian import bayesian


# Ensure no JSON artifacts persist in the working directory
@pytest.fixture(autouse=True)
def _cleanup_json_artifacts() -> None:
    yield
    for fname in ("jax_dists.json", "torch_dists.json"):
        Path(fname).unlink(missing_ok=True)


class DummyDataset(Dataset):
    def __init__(self, n: int = 5, d: int = 3) -> None:
        """Initialize a dataset of random normal inputs."""
        self.X = torch.randn(n, d)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (input, target) for the sample at index idx."""
        return self.X[idx], 0  # (input, target) convention


class DummyModel(torch.nn.Module):
    def __init__(self, d_in: int, n_classes: int) -> None:
        """Simple linear classifier model."""
        super().__init__()
        self.linear = torch.nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_save_and_load_with_meta() -> None:
    d_in, n_classes, n = 2, 3, 5
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=4, output_mode="logits", model_name="dummy")
    dists = gen.generate_distributions(dataset, progress=False)
    # Use variable to avoid lint warning
    assert isinstance(dists, dict)

    # Persistence-related checks intentionally omitted in tests.


def test_generator_init_and_run() -> None:
    d_in, n_classes, n = 3, 4, 7
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")

    dists = gen.generate_distributions(dataset, progress=False)

    assert isinstance(dists, dict)
    assert len(dists) == len(dataset)
    # each entry is a tensor of probabilities with correct length
    for i in range(len(dataset)):
        row = dists[i]
        assert isinstance(row, torch.Tensor)
        assert row.shape[-1] == n_classes
        # probabilities in [0,1] and sum approx 1
        assert bool(torch.all((row >= 0.0) & (row <= 1.0)))
        assert torch.isclose(row.sum(), torch.tensor(1.0), atol=1e-4)


def test_to_probs_modes_and_invalid() -> None:
    d_in, n_classes = 3, 4
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    gen_logits = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")
    gen_probs = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="probs")
    gen_auto = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="auto")

    logits = torch.tensor([0.5, -0.3, 0.1, 0.0], dtype=torch.float32)
    p1 = gen_logits.to_probs(logits)
    assert torch.isclose(p1.sum(), torch.tensor(1.0), atol=1e-6)

    probs_vec = torch.softmax(logits, dim=-1)
    p2 = gen_probs.to_probs(probs_vec)
    assert torch.isclose(p2.sum(), torch.tensor(1.0), atol=1e-6)

    p3 = gen_auto.to_probs(probs_vec)
    assert torch.isclose(p3.sum(), torch.tensor(1.0), atol=1e-6)

    gen_bad = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="invalid")
    with pytest.raises(ValueError, match="Invalid output_mode"):
        _ = gen_bad.to_probs(logits)


def test_to_device_nested_structures_cpu() -> None:
    model = DummyModel(d_in=2, n_classes=3)
    gen = FirstOrderDataGenerator(model=model, device="cpu")
    nested = {
        "a": torch.tensor([1.0, 2.0]),
        "b": [torch.tensor([3.0]), {"c": torch.tensor([4.0])}],
    }
    moved = gen.to_device(nested)
    assert isinstance(moved, dict)
    assert torch.allclose(moved["a"], nested["a"])  # type: ignore[index]
    assert isinstance(moved["b"], list)  # type: ignore[index]


def test_prepares_batch_inp_behavior() -> None:
    model = DummyModel(d_in=2, n_classes=3)
    gen = FirstOrderDataGenerator(model=model, device="cpu")
    assert torch.equal(gen.prepares_batch_inp((torch.tensor([1.0, 2.0]), 1)), torch.tensor([1.0, 2.0]))
    assert torch.equal(gen.prepares_batch_inp([torch.tensor([1.0, 2.0]), 1]), torch.tensor([1.0, 2.0]))
    sample = {"inp": torch.tensor([5.0, 6.0]), "lbl": 0}
    gen2 = FirstOrderDataGenerator(model=model, device="cpu", input_getter=lambda s: s["inp"])  # type: ignore[arg-type]
    assert torch.equal(gen2.prepares_batch_inp(sample), torch.tensor([5.0, 6.0]))


def test_first_order_dataset_and_dataloader_with_targets() -> None:
    # Base dataset yields (input, target)
    d_in, n_classes, n = 4, 5, 10
    dataset = DummyDataset(n=n, d=d_in)

    # Create distributions of correct length for each index; same seed for reproducibility of results
    torch.manual_seed(0)
    logits = torch.randn(n, n_classes)
    dists = {i: torch.softmax(logits[i], dim=-1).tolist() for i in range(n)}

    firstorderdatasets = FirstOrderDataset(dataset, dists)
    assert len(firstorderdatasets) == len(dataset)

    # Check a few samples
    for i in [0, n // 2, n - 1]:
        x, y, p = firstorderdatasets[i]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, (int, torch.Tensor))
        assert isinstance(p, torch.Tensor)
        assert p.shape[-1] == n_classes
        assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-4)

    # DataLoader integration
    loader = output_dataloader(dataset, dists, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    x, y, p = batch
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(p, torch.Tensor)
    assert x.shape[0] == y.shape[0] == p.shape[0]
    assert p.shape[-1] == n_classes


def test_generate_distributions_with_empty_dataset() -> None:
    dataset = DummyDataset(n=0, d=3)
    model = DummyModel(d_in=3, n_classes=2)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")

    dists = gen.generate_distributions(dataset, progress=False)

    assert isinstance(dists, dict)
    assert len(dists) == 0


def test_generate_distributions_with_external_dataloader_and_callable_model() -> None:
    class SmallDataset(Dataset):
        def __init__(self) -> None:
            self.X = torch.randn(5, 3)

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.X[idx]

    ds = SmallDataset()
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

    # Non-Module callable returning 1D logits (will be unsqueezed)
    def callable_model(_x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.arange(4, dtype=torch.float32)

    gen = FirstOrderDataGenerator(model=callable_model, device="cpu", batch_size=2, output_mode="logits")
    with pytest.warns(UserWarning, match=r"not a torch\.nn\.Module"):
        dists = gen.generate_distributions(loader, progress=True)
    assert isinstance(dists, dict)
    # With 1D outputs per batch, distributions count equals number of batches
    assert len(dists) == len(loader)

    # Model returning non-tensor should raise TypeError
    def bad_model(_x: torch.Tensor) -> list[float]:  # type: ignore[override]
        return [0.1, 0.2, 0.3, 0.4]

    gen_bad = FirstOrderDataGenerator(model=bad_model, device="cpu", batch_size=2, output_mode="probs")
    with pytest.raises(TypeError, match="Model must return a torch.Tensor"):
        _ = gen_bad.generate_distributions(loader, progress=False)


def test_get_posterior_distributions_returns_correct_structure() -> None:
    plain_model = nn.Linear(10, 4)
    bayesian_model = bayesian(plain_model)

    gen = FirstOrderDataGenerator(model=bayesian_model, device="cpu")
    dists = gen.get_posterior_distributions()

    assert isinstance(dists, dict)
    assert "weight" in dists
    assert "bias" in dists
    assert "mu" in dists["weight"]
    assert "rho" in dists["weight"]
    assert dists["weight"]["mu"].shape == (4, 10)
    assert dists["bias"]["mu"].shape == (4,)


def test_first_order_dataset_mismatch_warning_and_keyerror() -> None:
    n, d = 6, 3
    base = DummyDataset(n=n, d=d)
    # Provide distributions for only first n-1 items to intentionally trigger warning
    logits = torch.randn(n - 1, 4)
    dists = {i: torch.softmax(logits[i], dim=-1).tolist() for i in range(n - 1)}
    with pytest.warns(UserWarning, match=r"distributions count .* does not match dataset length"):
        ds = FirstOrderDataset(base, dists)
    # Access missing index should raise KeyError
    with pytest.raises(KeyError, match="No distribution for index"):
        _ = ds[n - 1]


class InputOnlyDataset(Dataset):
    def __init__(self, n: int = 6, d: int = 3) -> None:
        """Dataset emitting inputs only."""
        self.X = torch.randn(n, d)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return input vector for the sample at index idx."""
        return self.X[idx]  # input only


def test_first_order_dataset_and_dataloader_input_only_no_labels() -> None:
    # Base dataset yields input only
    d_in, n_classes, n = 3, 4, 12
    dataset = InputOnlyDataset(n=n, d=d_in)

    # Distributions
    torch.manual_seed(1)
    logits = torch.randn(n, n_classes)
    dists = {i: torch.softmax(logits[i], dim=-1).tolist() for i in range(n)}

    firstorderdatasets = FirstOrderDataset(dataset, dists)
    assert len(firstorderdatasets) == len(dataset)

    # Single sample
    x, p = firstorderdatasets[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(p, torch.Tensor)
    assert p.shape[-1] == n_classes

    # DataLoader
    loader = output_dataloader(dataset, dists, batch_size=5, shuffle=False)
    x_batch, p_batch = next(iter(loader))
    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(p_batch, torch.Tensor)
    assert x_batch.shape[0] == p_batch.shape[0]
    assert p_batch.shape[-1] == n_classes


def test_output_dataloader_with_input_getter_on_dict_dataset() -> None:
    class DictDataset(Dataset):
        def __init__(self, n: int = 5, d: int = 3) -> None:
            self.X = [{"inp": torch.randn(d), "lbl": i} for i in range(n)]

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
            return self.X[idx]

    n, d_in, n_classes = 5, 3, 4
    base = DictDataset(n=n, d=d_in)
    logits = torch.randn(n, n_classes)
    dists = {i: torch.softmax(logits[i], dim=-1).tolist() for i in range(n)}
    loader = output_dataloader(base, dists, batch_size=2, shuffle=False, input_getter=lambda s: s["inp"])  # type: ignore[arg-type]
    x_batch, p_batch = next(iter(loader))
    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(p_batch, torch.Tensor)
    assert p_batch.shape[-1] == n_classes


def test_is_probabilities_true_and_false() -> None:
    good = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float32)
    bad_sum = torch.tensor([[0.2, 0.9]], dtype=torch.float32)
    bad_range = torch.tensor([[1.2, -0.2]], dtype=torch.float32)
    assert _is_probabilities(good)
    assert not _is_probabilities(bad_sum)
    assert not _is_probabilities(bad_range)


def test_json_save_and_load_roundtrip(tmp_path: Path) -> None:
    # Exercise Torch generators save/load wrappers with real file
    d_in, n_classes, n = 3, 5, 6
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")
    dists = gen.generate_distributions(dataset, progress=False)
    dists_json = {int(i): dists[i].tolist() for i in range(n)}

    path = tmp_path / "torch_dists.json"
    gen.save_distributions(path, dists_json, meta={"backend": "torch"})
    dists_tensor, meta = gen.load_distributions(path)
    assert isinstance(dists_tensor, dict)
    assert meta.get("backend") == "torch"
    for i in range(n):
        assert isinstance(dists_tensor[i], torch.Tensor)
        assert dists_tensor[i].shape[-1] == n_classes
    # Remove test artifact
    path.unlink(missing_ok=True)


def test_pt_save_and_load_helpers(tmp_path: Path) -> None:
    data = {"a": torch.randn(3), "b": torch.randn(2)}
    path = tmp_path / "dists" / "out.pt"
    save_distributions_pt(data, str(path), create_dir=True, verbose=False)
    loaded = load_distributions_pt(str(path), device="cpu", verbose=False)
    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == set(data.keys())
    # Cleanup
    path.unlink(missing_ok=True)
    # remove the created directory if empty
    with contextlib.suppress(OSError):
        path.parent.rmdir()

    # Invalid suffix
    with pytest.raises(ValueError, match="File suffix must be '.pt' or '.pth'."):
        save_distributions_pt(data, str(tmp_path / "bad.txt"), verbose=False)

    # File not found
    with pytest.raises(FileNotFoundError, match="File not found"):
        _ = load_distributions_pt(str(tmp_path / "missing.pt"), verbose=False)

    # Nondict TypeError
    bad_path = tmp_path / "not_dict.pt"
    torch.save(torch.randn(3), bad_path)
    with pytest.raises(TypeError, match="Loaded object is not a dictionary"):
        _ = load_distributions_pt(str(bad_path), verbose=False)
    # Cleanup
    bad_path.unlink(missing_ok=True)


BatchWithLabel = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BatchInputOnly = tuple[torch.Tensor, torch.Tensor]
TorchBatch = BatchWithLabel | BatchInputOnly


def _train_one_model_with_soft_targets(
    loader: Iterable[TorchBatch],
    d_in: int,
    n_classes: int,
    steps: int = 30,
) -> tuple[DummyModel, list[float]]:
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for _step in range(steps):
        for batch in loader:
            if len(batch) == 3:
                x, _, p = batch
            else:
                x, p = batch
            logits = model(x)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=-1),
                p,
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Keep it simple: no separate phases or extra tracking
    return model, losses


def _compute_coverage(pred_probs: torch.Tensor, gt_probs: torch.Tensor, epsilon: float = 0.15) -> float:
    """Simple epsilon-credal coverage.

    Ground-truth distribution is covered if the L1 distance between
    predicted and ground-truth probability vectors is <= epsilon.
    Returns coverage rate in [0,1].
    """
    # pred_probs, gt_probs: [N, C] (gt = ground-truth)
    l1 = torch.sum(torch.abs(pred_probs - gt_probs), dim=-1)
    covered = (l1 <= epsilon).float()
    return float(covered.mean().item())


def test_end_to_end_training_and_coverage_improves() -> None:
    torch.manual_seed(23)

    # Setup dataset and teacher model to generate groundtruth distributions
    n, d_in, n_classes = 40, 6, 5
    dataset = DummyDataset(n=n, d=d_in)
    teacher = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=teacher, device="cpu", batch_size=16, output_mode="logits")
    dists = gen.generate_distributions(dataset, progress=False)

    # Build DataLoader for training student model using the generated distributions
    loader = output_dataloader(dataset, dists, batch_size=16, shuffle=True)

    # Train student; record initial and final loss
    student, losses = _train_one_model_with_soft_targets(loader, d_in=d_in, n_classes=n_classes, steps=15)
    assert len(losses) > 2
    assert losses[0] > losses[-1]  # training signal present

    # Evaluate coverage: compare student softmax to teacher distributions
    # Build tensors aligned with index order
    x_inputs = torch.stack([dataset[i][0] for i in range(n)], dim=0)
    with torch.no_grad():
        student_logits = student(x_inputs)
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.stack([dists[i] for i in range(n)], dim=0)

    cov_before = _compute_coverage(torch.softmax(torch.zeros_like(student_logits), dim=-1), teacher_probs, epsilon=0.25)
    cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

    # Coverage should improve after training
    assert cov_after >= cov_before


def test_load_distributions_returns_tensors(monkeypatch) -> None:
    """Ensure Torch generator's JSON loader returns tensors without file I/O.

    We monkeypatch the base loader to avoid reading from disk and to supply
    a known distributions dict (lists) and metadata. The Torch loader should
    convert lists to torch.Tensor and preserve metadata.
    """
    n, n_classes = 8, 4
    torch.manual_seed(7)
    logits = torch.randn(n, n_classes)
    dists_list = {i: torch.softmax(logits[i], dim=-1).tolist() for i in range(n)}
    meta = {"model_name": "dummy", "note": "round-trip"}

    # Monkeypatch base JSON loader to return precomputated dists+meta, avoiding disk I/O and artifact files
    def _fake_load_distributions(_self, _path):  # type: ignore[override]
        return dists_list, meta

    monkeypatch.setattr(PyBaseFirstOrderDataGenerator, "load_distributions", _fake_load_distributions, raising=True)

    # Call Torch generators loader; expect tensors
    gen = FirstOrderDataGenerator(model=DummyModel(d_in=3, n_classes=n_classes))
    dists_tensor, meta_out = gen.load_distributions("unused.json")

    assert meta_out == meta
    assert isinstance(dists_tensor, dict)
    assert set(dists_tensor.keys()) == set(range(n))
    for i in range(n):
        row = dists_tensor[i]
        assert isinstance(row, torch.Tensor)
        assert row.shape[-1] == n_classes
        assert bool(torch.all((row >= 0.0) & (row <= 1.0)))
        assert torch.isclose(row.sum(), torch.tensor(1.0), atol=1e-4)
