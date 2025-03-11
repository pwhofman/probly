from probly.utils import powerset, moebius
import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize

def upper_entropy(probs, base=2):
    def fun(x):
        return -entropy(x, base=base)
    x0 = probs.mean(axis=1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    ue = np.empty(probs.shape[0])
    for i in range(probs.shape[0]):
        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0)))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        ue[i] = -res.fun
    return ue

def lower_entropy(probs, base=2):
    def fun(x):
        return entropy(x, base=base)
    x0 = probs.mean(axis=1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    le = np.empty(probs.shape[0])
    for i in range(probs.shape[0]):
        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0)))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        le[i] = -res.fun
    return le

def generalised_hartley(probs, base=2):
    """
    Computes the generalised Hartley measure given the extreme points of
    a credal set
    outputs: array of shape (num_samples, num_members, num_classes)
    """
    gh = np.zeros(probs.shape[0])
    idxs = list(range(probs.shape[2]))  # list of class indices
    ps_A = powerset(idxs)  # powerset of all indices
    ps_A.pop(0)  # remove empty set
    for A in ps_A:
        m_A = moebius(probs, A)
        gh += m_A * (np.log(len(A)) / np.log(base))
    return gh
