from __future__ import annotations

from collections.abc import Iterable
import math
import random

from probly.data_generation.first_order_datagenerator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    output_dataloader,
)


class DummyDataset:
    def __init__(self, n: int = 5, d: int = 3) -> None:
        """Initialize a dataset with Gaussian inputs.

        n: number of samples; d: input dimension.
        """
        rng = random.Random(42)  # noqa: S311
        self.X: list[list[float]] = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        self.n = n
        self.d = d

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[list[float], int]:
        """Return (input, target) for the sample at index idx."""
        return self.X[idx], 0  # (input, target) convention


class InputOnlyDataset:
    def __init__(self, n: int = 6, d: int = 3) -> None:
        """Initialize an input-only dataset with Gaussian inputs."""
        rng = random.Random(43)  # noqa: S311
        self.X: list[list[float]] = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> list[float]:
        """Return input vector for the sample at index idx."""
        return self.X[idx]  # input only


def _softmax_row(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [0.0 for _ in row]


def _compute_coverage(pred_probs: list[list[float]], gt_probs: list[list[float]], epsilon: float = 0.15) -> float:
    """L1 epsilon coverage over lists of prob vectors."""
    assert len(pred_probs) == len(gt_probs)
    covered = 0
    for pp, qq in zip(pred_probs, gt_probs, strict=False):
        l1 = sum(abs(a - b) for a, b in zip(pp, qq, strict=False))
        if l1 <= epsilon:
            covered += 1
    return covered / max(1, len(gt_probs))


class DummyModel:
    """PurePython linear model producing logits for a batch of inputs.

    weights: [d_in][n_classes]
    forward(inputs) -> list of logits per sample
    """

    def __init__(self, d_in: int, n_classes: int, seed: int = 123) -> None:
        """Initialize a simple linear model with random weights."""
        rng = random.Random(seed)  # noqa: S311
        self.d_in = d_in
        self.n_classes = n_classes
        self.W: list[list[float]] = [[rng.uniform(-0.5, 0.5) for _ in range(n_classes)] for _ in range(d_in)]

    def __call__(self, inputs: list[list[float]]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for x in inputs:
            logits = [sum(x[j] * self.W[j][c] for j in range(self.d_in)) for c in range(self.n_classes)]
            outputs.append(logits)
        return outputs

    def predict_probs(self, inputs: list[list[float]]) -> list[list[float]]:
        return [_softmax_row(logits) for logits in self(inputs)]


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

    # (Removed inactive persistence test to keep test file clean.)


def test_first_order_dataset_and_dataloader_with_targets() -> None:
    d_in, n_classes, n = 4, 5, 10
    dataset = DummyDataset(n=n, d=d_in)

    rng = random.Random(0)  # noqa: S311
    logits = [[rng.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
    dists = {i: _softmax_row(logits[i]) for i in range(n)}

    firstorderdatasets = FirstOrderDataset(dataset, dists)
    assert len(firstorderdatasets) == len(dataset)

    # Check a few samples
    for i in [0, n // 2, n - 1]:
        x, y, p = firstorderdatasets[i]
        assert isinstance(x, list)
        assert isinstance(y, int)
        assert isinstance(p, list)
        assert len(p) == n_classes
        assert abs(sum(p) - 1.0) < 1e-4

    # DataLoader integration
    loader = output_dataloader(dataset, dists, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    # Batch is a list of samples; take the first
    x, y, p = batch[0]
    assert isinstance(x, list)
    assert isinstance(y, int)
    assert isinstance(p, list)
    assert len(p) == n_classes
    assert len(batch) <= 4  # first batch size


def test_generate_distributions_with_empty_dataset() -> None:
    dataset = DummyDataset(n=0, d=3)
    model = DummyModel(d_in=3, n_classes=2)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")

    dists = gen.generate_distributions(dataset, progress=False)

    assert isinstance(dists, dict)
    assert len(dists) == 0


def test_first_order_dataset_and_dataloader_input_only_no_labels() -> None:
    d_in, n_classes, n = 3, 4, 12
    dataset = InputOnlyDataset(n=n, d=d_in)

    rng = random.Random(1)  # noqa: S311
    logits = [[rng.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
    dists = {i: _softmax_row(logits[i]) for i in range(n)}

    firstorderdatasets = FirstOrderDataset(dataset, dists)
    assert len(firstorderdatasets) == len(dataset)

    # Single sample
    x, p = firstorderdatasets[0]
    assert isinstance(x, list)
    assert isinstance(p, list)
    assert len(p) == n_classes

    # DataLoader
    loader = output_dataloader(dataset, dists, batch_size=5, shuffle=False)
    batch = next(iter(loader))
    x0, p0 = batch[0]
    assert isinstance(x0, list)
    assert isinstance(p0, list)
    assert len(p0) == n_classes


SampleWithLabel = tuple[list[float], int, list[float]]
SampleInputOnly = tuple[list[float], list[float]]
Batch = list[SampleWithLabel | SampleInputOnly]


def _train_one_model_with_soft_targets(
    loader: Iterable[Batch],
    d_in: int,
    n_classes: int,
    steps: int = 30,
) -> tuple[DummyModel, list[float]]:
    model = DummyModel(d_in=d_in, n_classes=n_classes, seed=999)
    lr = 0.1
    losses: list[float] = []

    for _ in range(steps):
        for batch in loader:
            # batch is list of (x, y, p) or (x, p)
            for sample in batch:
                if len(sample) == 3:
                    x, _, q = sample
                else:
                    x, q = sample
                # forward
                logits = [sum(x[j] * model.W[j][c] for j in range(d_in)) for c in range(n_classes)]
                p = _softmax_row(logits)
                # KL(q || p) = sum q * (log q - log p)
                eps = 1e-12
                loss = sum(qc * (math.log(max(qc, eps)) - math.log(max(pc, eps))) for qc, pc in zip(q, p, strict=False))
                losses.append(loss)
                # Gradient scales ~ (p - q); weights update ~ x_j * (p_c - q_c)
                for c in range(n_classes):
                    delta = p[c] - q[c]
                    for j in range(d_in):
                        model.W[j][c] -= lr * delta * x[j]
    return model, losses


def test_end_to_end_training_and_coverage_improves() -> None:
    random.seed(23)

    n, d_in, n_classes = 40, 6, 5
    dataset = DummyDataset(n=n, d=d_in)
    teacher = DummyModel(d_in=d_in, n_classes=n_classes, seed=321)

    gen = FirstOrderDataGenerator(model=teacher, device="cpu", batch_size=16, output_mode="logits")
    dists = gen.generate_distributions(dataset, progress=False)

    # Build DataLoader for training student model using the generated distributions
    loader = output_dataloader(dataset, dists, batch_size=16, shuffle=True)

    # Train student; record initial and final loss
    student, losses = _train_one_model_with_soft_targets(loader, d_in=d_in, n_classes=n_classes, steps=15)
    assert len(losses) > 2
    assert losses[0] > losses[-1]

    # Evaluate coverage: compare student softmax to teacher distributions
    x_inputs = [dataset[i][0] for i in range(n)]
    student_probs = student.predict_probs(x_inputs)
    teacher_probs = [dists[i] for i in range(n)]

    uniform = [1.0 / n_classes] * n_classes
    cov_before = _compute_coverage([uniform for _ in range(n)], teacher_probs, epsilon=0.25)
    cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

    # Coverage should improve after training
    assert cov_after >= cov_before
