"""Conformal credal set prediction benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.method.conformal_credal_set_prediction._common import (
    conformal_dirichlet_relative_likelihood,
    conformal_inner_product,
    conformal_kullback_leibler,
    conformal_total_variation,
    conformal_wasserstein_distance,
)
from probly.method.evidential.classification import evidential_classification
from probly.quantification import quantify
from probly.representer import representer
from probly_benchmark.models import LeNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = LeNet(n_classes=5)
model.eval()
inputs = torch.randn(3, 1, 28, 28)
x_calib = torch.randn(10, 1, 28, 28)
y_calib = torch.randint(0, 5, (10,))
alpha = 0.1

logger.info("--- Total Variation ---")
cep_tv = conformal_total_variation(model)
cep_tv.calibrate(alpha, y_calib, x_calib)
rep_tv = representer(cep_tv)
prediction_tv = rep_tv.predict(inputs)
quant_tv = quantify(prediction_tv)
logger.info(quant_tv)

logger.info("--- Wasserstein Distance ---")
cep_wd = conformal_wasserstein_distance(model)
cep_wd.calibrate(alpha, y_calib, x_calib)
rep_wd = representer(cep_wd)
prediction_wd = rep_wd.predict(inputs)
quant_wd = quantify(prediction_wd)
logger.info(quant_wd)

logger.info("--- Inner Product ---")
cep_ip = conformal_inner_product(model)
cep_ip.calibrate(alpha, y_calib, x_calib)
rep_ip = representer(cep_ip)
prediction_ip = rep_ip.predict(inputs)
quant_ip = quantify(prediction_ip)
logger.info(quant_ip)

logger.info("--- Kullback-Leibler Divergence ---")
cep_kl = conformal_kullback_leibler(model)
cep_kl.calibrate(alpha, y_calib, x_calib)
rep_kl = representer(cep_kl)
prediction_kl = rep_kl.predict(inputs)
quant_kl = quantify(prediction_kl)
logger.info(quant_kl)

logger.info("--- Dirichlet Relative Likelihood ---")
evidential_model = evidential_classification(model)  # ty: ignore[invalid-argument-type]
dirichlet_cep = conformal_dirichlet_relative_likelihood(evidential_model)
dirichlet_cep.calibrate(alpha, y_calib, x_calib)
dirichlet_rep = representer(dirichlet_cep)
dirichlet_prediction = dirichlet_rep.predict(inputs)
dirichlet_quant = quantify(dirichlet_prediction)
logger.info(dirichlet_quant)
