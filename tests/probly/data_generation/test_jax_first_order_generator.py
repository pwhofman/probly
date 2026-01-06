from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import jax.numpy as jnp
import jax.nn as jnn
import pytest
import json

from probly.data_generation.jax_first_order_generator import (
	FirstOrderDataGenerator,
	FirstOrderDataset,
	output_dataloader,
)


class DummyDataset:
	def __init__(self, n: int = 5, d: int = 3):
		random.seed(42)
		self.X = [jnp.array([random.gauss(0, 1) for _ in range(d)], dtype=jnp.float32) for _ in range(n)]
		self.n = n
		self.d = d

	def __len__(self) -> int:
		return len(self.X)

	def __getitem__(self, idx: int):
		return self.X[idx], 0  # (input, target) convention


class InputOnlyDataset:
	def __init__(self, n: int = 6, d: int = 3):
		random.seed(43)
		self.X = [jnp.array([random.gauss(0, 1) for _ in range(d)], dtype=jnp.float32) for _ in range(n)]

	def __len__(self) -> int:
		return len(self.X)

	def __getitem__(self, idx: int):
		return self.X[idx]  # input only


def _softmax_row(row: List[float]) -> List[float]:
	arr = jnp.array(row, dtype=jnp.float32)
	p = jnn.softmax(arr, axis=-1)
	return [float(x) for x in p.tolist()]


class DummyModel:
	"""Simple linear model producing logits for a batch of inputs.

	W: [d_in, n_classes], b: [n_classes]
	forward(batch_inputs) -> [batch, n_classes]
	"""

	def __init__(self, d_in: int, n_classes: int, seed: int = 123):
		random.seed(seed)
		self.d_in = d_in
		self.n_classes = n_classes
		# Initialize small random weights
		self.W = jnp.array(
			[[random.uniform(-0.5, 0.5) for _ in range(n_classes)] for _ in range(d_in)], dtype=jnp.float32
		)
		self.b = jnp.zeros((n_classes,), dtype=jnp.float32)

	def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
		# inputs: [B, d_in] or [d_in] (broadcast-safe after ensure_2d)
		return jnp.dot(inputs, self.W) + self.b

	def predict_probs(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
		X = jnp.stack(inputs, axis=0)
		logits = self(X)
		return jnn.softmax(logits, axis=-1)


def test_generator_init_and_run(tmp_path: Path):
	d_in, n_classes, n = 3, 4, 7
	dataset = DummyDataset(n=n, d=d_in)
	model = DummyModel(d_in=d_in, n_classes=n_classes)

	gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=3, output_mode="logits")

	dists = gen.generate_distributions(dataset, progress=False)

	assert isinstance(dists, dict)
	assert len(dists) == len(dataset)
	for i in range(len(dataset)):
		row = dists[i]
		assert isinstance(row, list)
		assert len(row) == n_classes
		assert all(0.0 <= p <= 1.0 for p in row)
		assert abs(sum(row) - 1.0) < 1e-4


def test_first_order_dataset_and_dataloader_with_targets():
	d_in, n_classes, n = 4, 5, 10
	dataset = DummyDataset(n=n, d=d_in)

	random.seed(0)
	logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
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
	batch = next(iter(loader))
	x, y, p = batch[0]
	assert isinstance(x, jnp.ndarray) and isinstance(y, int) and isinstance(p, jnp.ndarray)
	assert p.shape[-1] == n_classes
	assert len(batch) <= 4


def test_generate_distributions_with_empty_dataset():
	dataset = DummyDataset(n=0, d=3)
	model = DummyModel(d_in=3, n_classes=2)
	gen = FirstOrderDataGenerator(model=model, device="cpu", batch_size=2, output_mode="logits")

	dists = gen.generate_distributions(dataset, progress=False)

	assert isinstance(dists, dict)
	assert len(dists) == 0


def test_first_order_dataset_and_dataloader_input_only_no_labels():
	d_in, n_classes, n = 3, 4, 12
	dataset = InputOnlyDataset(n=n, d=d_in)

	random.seed(1)
	logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n)]
	dists = {i: _softmax_row(logits[i]) for i in range(n)}

	firstorderdatasets = FirstOrderDataset(dataset, dists)
	assert len(firstorderdatasets) == len(dataset)

	x, p = firstorderdatasets[0]
	assert isinstance(x, jnp.ndarray)
	assert isinstance(p, jnp.ndarray)
	assert p.shape[-1] == n_classes

	loader = output_dataloader(dataset, dists, batch_size=5, shuffle=False)
	batch = next(iter(loader))
	x0, p0 = batch[0]
	assert isinstance(x0, jnp.ndarray) and isinstance(p0, jnp.ndarray)
	assert p0.shape[-1] == n_classes


def _train_one_model_with_soft_targets(loader, d_in: int, n_classes: int, steps: int = 30):
	model = DummyModel(d_in=d_in, n_classes=n_classes, seed=999)
	lr = 0.1
	losses: List[float] = []

	for _ in range(steps):
		for batch in loader:
			# batch is list of (x, y, p) or (x, p)
			for sample in batch:
				if len(sample) == 3:
					x, _, q = sample
				else:
					x, q = sample
				# forward
				logits = jnp.dot(x, model.W) + model.b  # [C]
				p = jnn.softmax(logits, axis=-1)
				# KL(q || p) = sum q * (log q - log p)
				eps = 1e-12
				loss = jnp.sum(q * (jnp.log(jnp.maximum(q, eps)) - jnp.log(jnp.maximum(p, eps))))
				losses.append(float(loss))
				# how the loss changes when you nudge the logits: (p - q); weight update: x_j * (p_c - q_c)
				delta = (p - q)  # [C]
				grad_W = jnp.outer(x, delta)  # [d_in, C]
				grad_b = delta  # [C]
				model.W = model.W - lr * grad_W
				model.b = model.b - lr * grad_b
	return model, losses


def _compute_coverage(pred_probs: jnp.ndarray, gt_probs: jnp.ndarray, epsilon: float = 0.15) -> float:
	"""Epsilon-credal coverage: L1 distance <= epsilon means covered."""
	l1 = jnp.sum(jnp.abs(pred_probs - gt_probs), axis=-1)
	covered = (l1 <= epsilon).astype(jnp.float32)
	return float(jnp.mean(covered))


def test_end_to_end_training_and_coverage_improves():
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

	X = [dataset[i][0] for i in range(n)]
	student_probs = student.predict_probs(X)
	teacher_probs = jnp.array([dists[i] for i in range(n)], dtype=jnp.float32)

	uniform = jnp.full((n_classes,), 1.0 / n_classes, dtype=jnp.float32)
	cov_before = _compute_coverage(jnp.tile(uniform, (n, 1)), teacher_probs, epsilon=0.25)
	cov_after = _compute_coverage(student_probs, teacher_probs, epsilon=0.25)

	# Deactivated to avoid writing run logs during tests:
	# # Visual feedback: save a small JSON run log
	# repo_root = Path(__file__).resolve().parents[3]
	# datagen_dir = repo_root / "src" / "probly" / "data_generation"
	# datagen_dir.mkdir(parents=True, exist_ok=True)
	# run_log = {
	# 	"run_summary_docs": {
	# 		"n": "number of samples in the dataset used",
	# 		"d_in": "input feature dimension",
	# 		"n_classes": "number of output classes",
	# 		"loss_initial": "first recorded training loss (KL divergence) at the start; higher means the student disagrees more with the distributions",
	# 		"loss_final": "last recorded training loss; lower than initial indicates learning happened",
	# 		"coverage_before": "baseline agreement rate using a naive uniform prediction vs teacher distributions, measured by L1 distance ≤ epsilon (e.g., 0.25)",
	# 		"coverage_after": "agreement rate after training using the students predictions; higher than before shows improvement",
	# 	},
	# 	"n": n,
	# 	"d_in": d_in,
	# 	"n_classes": n_classes,
	# 	"loss_initial": float(losses[0]),
	# 	"loss_final": float(losses[-1]),
	# 	"coverage_before": float(cov_before),
	# 	"coverage_after": float(cov_after),
	# }
	# log_path = datagen_dir / "run_summary.json"
	# with open(log_path, "w", encoding="utf-8") as f:
	# 	json.dump(run_log, f, ensure_ascii=False, indent=2)

	assert cov_after >= cov_before
