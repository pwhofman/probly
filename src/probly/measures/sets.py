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
