"""Benchmark for Dirichlet calibration."""

from __future__ import annotations

import logging

import torch

from probly.calibrator import calibrate
from probly.method.dirichlet_calibration import dirichlet_calibration
from probly.predictor import predict_raw
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_CLASSES = 5

model = LeNet(n_classes=N_CLASSES)
calibrator = dirichlet_calibration(model, num_classes=N_CLASSES, predictor_type="logit_classifier")

calib_inputs = torch.randn(64, 1, 28, 28)
calib_labels = torch.randint(0, N_CLASSES, (64,))
calibrate(calibrator, calib_labels, calib_inputs)

inputs = torch.randn(3, 1, 28, 28)
output = predict_raw(calibrator, inputs)
logger.info(output)
logger.info(output.shape)
