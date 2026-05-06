"""Conformal credal set prediction benchmark code."""

from __future__ import annotations

import logging

import torch

from probly.method.conformal_credal_set import (
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
evidential_model = evidential_classification(model)
dirichlet_cep = conformal_dirichlet_relative_likelihood(evidential_model)
dirichlet_cep.calibrate(alpha, y_calib, x_calib)
dirichlet_rep = representer(dirichlet_cep)
dirichlet_prediction = dirichlet_rep.predict(inputs)
dirichlet_quant = quantify(dirichlet_prediction)
logger.info(dirichlet_quant)


# Same four credal-set methods exercised with first-order calibration data
# (probability vectors instead of class indices).
y_calib_first_order = torch.softmax(torch.randn(10, 5), dim=-1)

logger.info("--- Total Variation (first-order calibration) ---")
cep_tv_fo = conformal_total_variation(LeNet(n_classes=5).eval())
cep_tv_fo.calibrate(alpha, y_calib_first_order, x_calib)
rep_tv_fo = representer(cep_tv_fo)
quant_tv_fo = quantify(rep_tv_fo.predict(inputs))
logger.info(quant_tv_fo)

logger.info("--- Wasserstein Distance (first-order calibration) ---")
cep_wd_fo = conformal_wasserstein_distance(LeNet(n_classes=5).eval())
cep_wd_fo.calibrate(alpha, y_calib_first_order, x_calib)
rep_wd_fo = representer(cep_wd_fo)
quant_wd_fo = quantify(rep_wd_fo.predict(inputs))
logger.info(quant_wd_fo)

logger.info("--- Inner Product (first-order calibration) ---")
cep_ip_fo = conformal_inner_product(LeNet(n_classes=5).eval())
cep_ip_fo.calibrate(alpha, y_calib_first_order, x_calib)
rep_ip_fo = representer(cep_ip_fo)
quant_ip_fo = quantify(rep_ip_fo.predict(inputs))
logger.info(quant_ip_fo)

logger.info("--- Kullback-Leibler Divergence (first-order calibration) ---")
cep_kl_fo = conformal_kullback_leibler(LeNet(n_classes=5).eval())
cep_kl_fo.calibrate(alpha, y_calib_first_order, x_calib)
rep_kl_fo = representer(cep_kl_fo)
quant_kl_fo = quantify(rep_kl_fo.predict(inputs))
logger.info(quant_kl_fo)
