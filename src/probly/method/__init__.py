"""Uncertainty-aware methods."""

from probly.method.batchensemble import batchensemble
from probly.method.bayesian import bayesian
from probly.method.cast import cast
from probly.method.conformal_credal_set import (
    conformal_dirichlet_relative_likelihood,
    conformal_inner_product,
    conformal_kullback_leibler,
    conformal_total_variation,
    conformal_wasserstein_distance,
)
from probly.method.credal_bnn import credal_bnn
from probly.method.credal_ensembling import credal_ensembling
from probly.method.credal_relative_likelihood import credal_relative_likelihood
from probly.method.credal_wrapper import credal_wrapper
from probly.method.dare import dare
from probly.method.ddu import ddu
from probly.method.dropconnect import dropconnect
from probly.method.dropout import dropout
from probly.method.duq import duq
from probly.method.efficient_credal_prediction import efficient_credal_prediction
from probly.method.ensemble import ensemble
from probly.method.evidential.classification import evidential_classification
from probly.method.evidential.regression import evidential_regression
from probly.method.g_vbll import g_vbll
from probly.method.graph_posterior_network import (
    cuq_graph_neural_network,
    graph_posterior_network,
    lop_graph_posterior_network,
)
from probly.method.het_net import het_net
import probly.method.laplace  # noqa: F401  # registers LaplaceRepresenter lazily
from probly.method.mahalanobis import mahalanobis
from probly.method.sngp import sngp
from probly.method.subensemble import subensemble
from probly.method.vbll import vbll

__all__ = [
    "batchensemble",
    "bayesian",
    "cast",
    "conformal_dirichlet_relative_likelihood",
    "conformal_inner_product",
    "conformal_kullback_leibler",
    "conformal_total_variation",
    "conformal_wasserstein_distance",
    "credal_bnn",
    "credal_ensembling",
    "credal_relative_likelihood",
    "credal_wrapper",
    "cuq_graph_neural_network",
    "dare",
    "ddu",
    "dropconnect",
    "dropout",
    "duq",
    "efficient_credal_prediction",
    "ensemble",
    "evidential_classification",
    "evidential_regression",
    "g_vbll",
    "graph_posterior_network",
    "het_net",
    "lop_graph_posterior_network",
    "mahalanobis",
    "sngp",
    "subensemble",
    "vbll",
]
