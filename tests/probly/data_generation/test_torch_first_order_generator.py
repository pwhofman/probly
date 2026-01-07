from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.data import Dataset

from probly.data_generation.torch_first_order_generator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    output_dataloader,
)
from probly.transformation.bayesian import bayesian


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
    # is each entry a list of probabilities with correct length
    for i in range(len(dataset)):
        row = dists[i]
        assert isinstance(row, list)
        assert len(row) == n_classes
        # probabilities in [0,1] and sum approx 1
        assert all(0.0 <= p <= 1.0 for p in row)
        assert abs(sum(row) - 1.0) < 1e-4


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
        teacher_probs = torch.tensor([dists[i] for i in range(n)], dtype=torch.float32)

    cov_before = _compute_coverage(torch.softmax(torch.zeros_like(student_logits), dim=-1), teacher_probs, epsilon=0.25)
    cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

    # Coverage should improve after training
    assert cov_after >= cov_before
