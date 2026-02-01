from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import random

try:
    import jax
    import jax.nn as jnn
    import jax.numpy as jnp
except ModuleNotFoundError:
    import pytest

    pytest.skip(
        "JAX not installed so skipping JAX-dependent tests.",
        allow_module_level=True,
    )
import pytest

from probly.data_generation.first_order_datagenerator import SimpleDataLoader
from probly.data_generation.jax_first_order_generator import (
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
        self.X = [jnp.array([rng.gauss(0, 1) for _ in range(d)], dtype=jnp.float32) for _ in range(n)]
        self.n = n
        self.d = d

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, int]:
        """Return (input, target) for the sample at index idx."""
        return self.X[idx], 0  # (input, target) convention


class InputOnlyDataset:
    def __init__(self, n: int = 6, d: int = 3) -> None:
        """Initialize an input-only dataset with Gaussian inputs."""
        rng = random.Random(43)  # noqa: S311
        self.X = [jnp.array([rng.gauss(0, 1) for _ in range(d)], dtype=jnp.float32) for _ in range(n)]

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> jnp.ndarray:
        """Return input vector for the sample at index idx."""
        return self.X[idx]  # input only


def _softmax_row(row: list[float]) -> list[float]:
    arr = jnp.array(row, dtype=jnp.float32)
    p = jnn.softmax(arr, axis=-1)
    return [float(x) for x in p.tolist()]


class DummyModel:
    """Simple linear model producing logits for a batch of inputs.

    W: [d_in, n_classes], b: [n_classes]
    forward(batch_inputs) -> [batch, n_classes]
    """

    def __init__(self, d_in: int, n_classes: int, seed: int = 123) -> None:
        """Initialize a simple linear model with random weights."""
        rng = random.Random(seed)  # noqa: S311
        self.d_in = d_in
        self.n_classes = n_classes
        # Initialize small random weights
        self.W = jnp.array(
            [[rng.uniform(-0.5, 0.5) for _ in range(n_classes)] for _ in range(d_in)],
            dtype=jnp.float32,
        )
        self.b = jnp.zeros((n_classes,), dtype=jnp.float32)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: [B, d_in] or [d_in] (broadcast-safe after ensure_2d)
        return jnp.dot(inputs, self.W) + self.b

    def predict_probs(self, inputs: list[jnp.ndarray]) -> jnp.ndarray:
        x_stack = jnp.stack(inputs, axis=0)
        logits = self(x_stack)
        return jnn.softmax(logits, axis=-1)


def test_generator_init_and_run() -> None:
    d_in, n_classes, n = 3, 4, 7
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)

    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")

    dists = gen.generate_distributions(dataset, progress=False)

    assert isinstance(dists, dict)
    assert len(dists) == len(dataset)
    for i in range(len(dataset)):
        row = dists[i]
        assert isinstance(row, jnp.ndarray)
        assert row.shape[-1] == n_classes
        assert bool(jnp.all((row >= 0.0) & (row <= 1.0)))
        assert float(jnp.sum(row)) == pytest.approx(1.0, abs=1e-4)


def test_first_order_dataset_and_dataloader_with_targets() -> None:
    d_in, n_classes, n = 4, 5, 10
    dataset = DummyDataset(n=n, d=d_in)

    rng = random.Random(0)  # noqa: S311
    logits = [[rng.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
    dists = {i: _softmax_row(logits[i]) for i in range(n)}

    firstorderdatasets = FirstOrderDataset(dataset, dists)
    assert len(firstorderdatasets) == len(dataset)

    for i in [0, n // 2, n - 1]:
        x, y, p = firstorderdatasets[i]
        assert isinstance(x, jnp.ndarray)
        assert isinstance(y, int)
        assert isinstance(p, jnp.ndarray)
        assert p.shape[-1] == n_classes
        assert float(jnp.sum(p)) == pytest.approx(1.0, abs=1e-4)

    loader = output_dataloader(dataset, dists, batch_size=4, shuffle=False)
    inpt_batch, lbl_batch, dist_batch = next(iter(loader))
    assert isinstance(inpt_batch, jnp.ndarray)
    assert isinstance(dist_batch, jnp.ndarray)
    assert dist_batch.shape[-1] == n_classes
    # Inspect first sample
    x, y, p = inpt_batch[0], int(lbl_batch[0]), dist_batch[0]
    assert isinstance(x, jnp.ndarray)
    assert isinstance(y, int)
    assert isinstance(p, jnp.ndarray)
    assert p.shape[-1] == n_classes


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

    x, p = firstorderdatasets[0]
    assert isinstance(x, jnp.ndarray)
    assert isinstance(p, jnp.ndarray)
    assert p.shape[-1] == n_classes

    loader = output_dataloader(dataset, dists, batch_size=5, shuffle=False)
    inpt_batch, dist_batch = next(iter(loader))
    x0, p0 = inpt_batch[0], dist_batch[0]
    assert isinstance(x0, jnp.ndarray)
    assert isinstance(p0, jnp.ndarray)
    assert p0.shape[-1] == n_classes


SampleWithLabel = tuple[jnp.ndarray, int, jnp.ndarray]
SampleInputOnly = tuple[jnp.ndarray, jnp.ndarray]
Batch = list[SampleWithLabel | SampleInputOnly]


def _train_one_model_with_soft_targets(
    loader: Iterable[Batch] | Iterable[object],
    d_in: int,
    n_classes: int,
    steps: int = 30,
) -> tuple[DummyModel, list[float]]:
    model = DummyModel(d_in=d_in, n_classes=n_classes, seed=999)
    lr = 0.1
    losses: list[float] = []

    @jax.jit
    def loss_fn(W: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, q: jnp.ndarray) -> jnp.float32:
        logits = jnp.dot(x, W) + b  # [C]
        p = jnn.softmax(logits, axis=-1)
        eps = 1e-12
        return jnp.sum(q * (jnp.log(jnp.maximum(q, eps)) - jnp.log(jnp.maximum(p, eps))))

    grad_loss = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    for _ in range(steps):
        for batch in loader:
            # Support both list-of-samples and batched JAX arrays
            if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], (list, tuple)):
                # Old format: list of samples
                for sample in batch:
                    if len(sample) == 3:
                        x, _, q = sample
                    else:
                        x, q = sample
                    (loss_val, (gw, gb)) = grad_loss(model.W, model.b, x, q)
                    losses.append(float(loss_val))
                    model.W = model.W - lr * gw
                    model.b = model.b - lr * gb
            # New format: batched arrays
            elif isinstance(batch, tuple) and len(batch) == 3:
                inpt_batch, _lbl_batch, dist_batch = batch
                for i in range(inpt_batch.shape[0]):
                    x, q = inpt_batch[i], dist_batch[i]
                    (loss_val, (gw, gb)) = grad_loss(model.W, model.b, x, q)
                    losses.append(float(loss_val))
                    model.W = model.W - lr * gw
                    model.b = model.b - lr * gb
            else:
                inpt_batch, dist_batch = batch
                for i in range(inpt_batch.shape[0]):
                    x, q = inpt_batch[i], dist_batch[i]
                    (loss_val, (gw, gb)) = grad_loss(model.W, model.b, x, q)
                    losses.append(float(loss_val))
                    model.W = model.W - lr * gw
                    model.b = model.b - lr * gb
    return model, losses


def _compute_coverage(pred_probs: jnp.ndarray, gt_probs: jnp.ndarray, epsilon: float = 0.15) -> float:
    """Epsilon-credal coverage: L1 distance <= epsilon means covered."""
    l1 = jnp.sum(jnp.abs(pred_probs - gt_probs), axis=-1)
    covered = (l1 <= epsilon).astype(jnp.float32)
    return float(jnp.mean(covered))


def test_end_to_end_training_and_coverage_improves() -> None:
    random.seed(23)

    n, d_in, n_classes = 40, 6, 5
    dataset = DummyDataset(n=n, d=d_in)
    teacher = DummyModel(d_in=d_in, n_classes=n_classes, seed=321)

    gen = FirstOrderDataGenerator(model=teacher, device="cpu", batch_size=16, output_mode="logits")
    dists = gen.generate_distributions(dataset, progress=False)

    loader = output_dataloader(dataset, dists, batch_size=16, shuffle=True)

    student, losses = _train_one_model_with_soft_targets(loader, d_in=d_in, n_classes=n_classes, steps=15)
    assert len(losses) > 2
    assert losses[0] > losses[-1]

    x_inputs = [dataset[i][0] for i in range(n)]
    student_probs = student.predict_probs(x_inputs)
    teacher_probs = jnp.array([dists[i] for i in range(n)], dtype=jnp.float32)

    uniform = jnp.full((n_classes,), 1.0 / n_classes, dtype=jnp.float32)
    cov_before = _compute_coverage(jnp.tile(uniform, (n, 1)), teacher_probs, epsilon=0.25)
    cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

    assert cov_after >= cov_before


def test_to_probs_variants_and_transform_and_return_jax_false() -> None:
    d_in, n_classes = 3, 4
    model = DummyModel(d_in=d_in, n_classes=n_classes)

    logits = jnp.array([0.5, -0.3, 0.1, 0.0], dtype=jnp.float32)
    probs_vec = jnn.softmax(logits, axis=-1)

    # logits mode applies softmax
    gen_logits = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")
    p1 = gen_logits.to_probs(logits)
    assert isinstance(p1, jnp.ndarray)
    assert float(jnp.sum(p1)) == pytest.approx(1.0, abs=1e-6)

    # probs mode passes through
    gen_probs = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="probs")
    p2 = gen_probs.to_probs(probs_vec)
    assert isinstance(p2, jnp.ndarray)
    assert float(jnp.sum(p2)) == pytest.approx(1.0, abs=1e-6)

    # auto detects probabilities
    gen_auto = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="auto")
    p3 = gen_auto.to_probs(probs_vec)
    assert isinstance(p3, jnp.ndarray)
    assert float(jnp.sum(p3)) == pytest.approx(1.0, abs=1e-6)

    # output_transform overrides output_mode
    gen_xform = FirstOrderDataGenerator(
        model=model,
        device="cpu",
        batch_size=2,
        output_mode="probs",
        output_transform=lambda out: jnn.softmax(jnp.asarray(out), axis=-1),
    )
    p4 = gen_xform.to_probs(logits)
    assert isinstance(p4, jnp.ndarray)
    assert float(jnp.sum(p4)) == pytest.approx(1.0, abs=1e-6)

    # invalid output_mode raises
    gen_bad = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="invalid")
    with pytest.raises(ValueError, match="Invalid output_mode"):
        _ = gen_bad.to_probs(logits)

    # return_jax=False yields list-of-lists
    gen_list = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="probs", return_jax=False)
    out_list = gen_list.to_probs(probs_vec)
    assert isinstance(out_list, list)
    assert isinstance(out_list[0], list)
    assert pytest.approx(sum(out_list[0]), abs=1e-6) == 1.0

    # None outputs path produces empty array
    gen_none = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="auto")
    empty = gen_none.to_probs(None)
    assert isinstance(empty, jnp.ndarray)
    assert empty.shape == (0, 0)


def test_to_device_nested_structures_cpu() -> None:
    d_in, n_classes = 2, 3
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=1, output_mode="probs")

    nested = {
        "a": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "b": [jnp.array([3.0], dtype=jnp.float32), {"c": jnp.array([4.0], dtype=jnp.float32)}],
    }
    moved = gen.to_device(nested)
    # Types preserved and values equal
    assert isinstance(moved, dict)
    assert jnp.allclose(moved["a"], nested["a"])  # type: ignore[index]
    assert isinstance(moved["b"], list)  # type: ignore[index]


def test_generate_distributions_with_simpledataloader_progress() -> None:
    d_in, n_classes, n = 3, 4, 9
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")

    loader = SimpleDataLoader(dataset, batch_size=3, shuffle=False)
    dists = gen.generate_distributions(loader, progress=True)
    assert isinstance(dists, dict)
    assert len(dists) == n


class RaggedWeirdDataset:
    def __init__(self, n: int = 7) -> None:
        """Initialize a dataset with ragged JAX inputs and dict labels."""
        self.n = n

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n

    def __getitem__(self, idx: int) -> tuple[dict[str, object], dict[str, int]]:
        """Return (input_dict, label_dict) for index idx."""
        # Ragged input lengths and dict labels to force fallbacks
        x_len = (idx % 3) + 1
        x = jnp.array([float(idx + 1)] * x_len, dtype=jnp.float32)
        return {"id": idx, "x": x}, {"label": idx}


def test_jax_output_dataloader_ragged_inputs_and_shuffle_determinism() -> None:
    n_classes, n = 4, 8
    base = RaggedWeirdDataset(n=n)

    rng = random.Random(7)  # noqa: S311
    logits = [[rng.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
    dists = {i: [float(x) for x in jnn.softmax(jnp.array(logits[i], dtype=jnp.float32)).tolist()] for i in range(n)}

    loader1 = output_dataloader(base, dists, batch_size=3, shuffle=True, seed=0, device="cpu")
    inpt_batch, lbl_batch, dist_batch = next(iter(loader1))
    # Inputs and labels fall back to lists due to ragged shapes/dict labels
    assert isinstance(inpt_batch, list)
    assert isinstance(inpt_batch[0], dict)
    assert isinstance(lbl_batch, list)
    assert isinstance(dist_batch, jnp.ndarray)

    # Deterministic with same seed
    loader2 = output_dataloader(base, dists, batch_size=3, shuffle=True, seed=0, device="cpu")
    inpt_batch2, lbl_batch2, dist_batch2 = next(iter(loader2))
    ids1 = [sample["id"] for sample in inpt_batch]
    ids2 = [sample["id"] for sample in inpt_batch2]
    assert ids1 == ids2
    assert jnp.array_equal(dist_batch, dist_batch2)

    # Different seed should change order
    loader3 = output_dataloader(base, dists, batch_size=3, shuffle=True, seed=123, device="cpu")
    inpt_batch3, lbl_batch3, dist_batch3 = next(iter(loader3))
    ids3 = [sample["id"] for sample in inpt_batch3]
    assert ids1 != ids3


def test_save_and_load_distributions_jax_conversion(tmp_path: Path) -> None:
    """Ensure save/load returns JAX arrays and meta roundtrips."""
    d_in, n_classes, n = 3, 5, 6
    dataset = DummyDataset(n=n, d=d_in)
    model = DummyModel(d_in=d_in, n_classes=n_classes)
    gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")

    # Build distributions frmo teacher
    dists = gen.generate_distributions(dataset, progress=False)
    # Convert to JSON friendly lists
    dists_json = {int(i): [float(x) for x in jnp.array(dists[i]).tolist()] for i in range(n)}

    # Save and load using pathlib
    path = tmp_path / "jax_dists.json"
    gen.save_distributions(path, dists_json, meta={"backend": "jax"})

    jax_dists, meta = gen.load_distributions(path)
    assert isinstance(jax_dists, dict)
    assert isinstance(meta, dict)
    assert meta.get("backend") == "jax"
    for i in range(n):
        assert isinstance(jax_dists[i], jnp.ndarray)
        assert jax_dists[i].shape[-1] == n_classes
