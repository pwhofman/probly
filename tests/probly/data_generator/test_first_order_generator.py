# command i use: pytest "c:\Users\ashhe\Desktop\Informatik mit integriertem Anwendungsfach\5.Semester\WP14 SEP Probly\PythonProjekte\probly\tests\probly\data_generator\test_first_order_generator.py" -q
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from probly.data_generator.first_order_generator import (
    FirstOrderDataGenerator,
    FirstOrderDataset,
    output_fo_dataloader,
)


class DummyDataset(Dataset):
    def __init__(self, n=5, d=3):
        self.X = torch.randn(n, d)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], 0  # (input, target) convention


class DummyModel(torch.nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_save_and_load_with_meta(tmp_path: Path):
    d_in, n_classes, n = 2, 3, 5
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=4, output_mode="logits", model_name="dummy")
    dists = gen.generate_distributions(dataset, progress=False)

    # Persist file into repository under src/probly/data_generator
    repo_root = Path(__file__).resolve().parents[3]
    save_path = repo_root / "src" / "probly" / "data_generator" / "first_order_dists.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"note": "unit-test", "classes": n_classes}

    gen.save_distributions(save_path, dists, meta=meta)

    # basic file integrity
    with open(save_path, encoding="utf-8") as f:
        obj = json.load(f)
    assert "distributions" in obj
    assert "meta" in obj

    loaded_dists, loaded_meta = gen.load_distributions(save_path)

    assert loaded_dists == dists
    assert loaded_meta.get("model_name") == "dummy"
    assert loaded_meta.get("note") == "unit-test"
    assert loaded_meta.get("classes") == n_classes


def test_generator_init_and_run(tmp_path: Path):
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


def test_first_order_dataset_and_dataloader_with_targets():
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
    loader = output_fo_dataloader(dataset, dists, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    x, y, p = batch
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(p, torch.Tensor)
    assert x.shape[0] == y.shape[0] == p.shape[0]
    assert p.shape[-1] == n_classes


def test_generate_distributions_with_empty_dataset():
    dataset = DummyDataset(n=0, d=3)
    model = DummyModel(d_in=3, n_classes=2)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")

    dists = gen.generate_distributions(dataset, progress=False)

    assert isinstance(dists, dict)
    assert len(dists) == 0


def test_get_posterior_distributions_returns_correct_structure():
    from torch import nn

    from probly.transformation.bayesian import bayesian

    plain_model = nn.Linear(10, 4)
    bayesian_model = bayesian(plain_model)

    gen = FirstOrderDataGenerator(model=bayesian_model, device="cpu")
    dists = gen.get_posterior_distributions()

    assert isinstance(dists, dict)
    assert "weight" in dists and "bias" in dists
    assert "mu" in dists["weight"] and "rho" in dists["weight"]
    assert dists["weight"]["mu"].shape == (4, 10)
    assert dists["bias"]["mu"].shape == (4,)


class InputOnlyDataset(Dataset):
    def __init__(self, n=6, d=3):
        self.X = torch.randn(n, d)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]  # input only


def test_first_order_dataset_and_dataloader_input_only_no_labels():
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
    loader = output_fo_dataloader(dataset, dists, batch_size=5, shuffle=False)
    x_batch, p_batch = next(iter(loader))
    assert isinstance(x_batch, torch.Tensor) and isinstance(p_batch, torch.Tensor)
    assert x_batch.shape[0] == p_batch.shape[0]
    assert p_batch.shape[-1] == n_classes


def _train_one_model_with_soft_targets(loader, d_in: int, n_classes: int, steps: int = 30):
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = []
    for step in range(steps):
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
    """Simple epsilon-credal coverage: groundtruth distribution covered if
    L1 distance between predicted prob vector and groundtruth prob vector <= epsilon.
    Returns coverage rate in [0,1].
    """
    # pred_probs, gt_probs: [N, C] (gt = ground-truth)
    l1 = torch.sum(torch.abs(pred_probs - gt_probs), dim=-1)
    covered = (l1 <= epsilon).float()
    # covered = tensor of 0.0/1.0 covered.mean = mean over all samples(coveragerate as skalar tensor) covered.mean.item = skalartensor -> pythonfloat
    return covered.mean().item()


def test_end_to_end_training_and_coverage_improves():
    torch.manual_seed(23)

    # Setup dataset and teacher model to generate groundtruth distributions
    n, d_in, n_classes = 40, 6, 5
    dataset = DummyDataset(n=n, d=d_in)
    teacher = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=teacher, device="cpu", batch_size=16, output_mode="logits")
    dists = gen.generate_distributions(dataset, progress=False)

    # Build DataLoader for training student model using the generated distributions
    loader = output_fo_dataloader(dataset, dists, batch_size=16, shuffle=True)

    # Train student; record initial and final loss
    student, losses = _train_one_model_with_soft_targets(loader, d_in=d_in, n_classes=n_classes, steps=15)
    assert len(losses) > 2
    assert losses[0] > losses[-1]  # training signal present

    # Visual feedback: save model and a small JSON run log
    repo_root = Path(__file__).resolve().parents[3]
    datagen_dir = repo_root / "src" / "probly" / "data_generator"
    datagen_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate coverage: compare student softmax to teacher distributions
    # Build tensors aligned with index order
    X = torch.stack([dataset[i][0] for i in range(n)], dim=0)
    with torch.no_grad():
        student_logits = student(X)
        student_probs = torch.softmax(student_logits, dim=-1)
        teacher_probs = torch.tensor([dists[i] for i in range(n)], dtype=torch.float32)

    cov_before = _compute_coverage(torch.softmax(torch.zeros_like(student_logits), dim=-1), teacher_probs, epsilon=0.25)
    cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

    # Write run summary JSON for verification
    run_log = {
        "run_summary_docs": {
            "n": "number of samples in the dataset used",
            "d_in": "input feature dimension",
            "n_classes": "number of output classes",
            "loss_initial": "first recorded training loss (KL divergence) at the start; higher means the student disagrees more with the distributions",
            "loss_final": "last recorded training loss; lower than initial indicates learning happened",
            "coverage_before": "baseline agreement rate using a naive uniform prediction vs teacher distributions, measured by L1 distance ≤ epsilon (e.g., 0.25)",
            "coverage_after": "agreement rate after training using the students predictions; higher than before shows improvement",
        },
        "n": n,
        "d_in": d_in,
        "n_classes": n_classes,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "coverage_before": cov_before,
        "coverage_after": cov_after,
    }
    log_path = datagen_dir / "run_summary.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    # Minimal console cue
    print(f"Saved run log to: {log_path}")

    # Coverage should improve after training
    assert cov_after >= cov_before
