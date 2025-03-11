import itertools
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))

def capacity(Q, A):
    """Computes the capacity of a set Q given a set A"""
    sum = np.sum(Q[:, :, A], axis=2)
    min = np.min(sum, axis=1)
    return min

def moebius(Q, A):
    """Computes the Moebius function of a set Q given a set A.
    Q: array of shape (num_samples, num_members, num_classes)
    A: set of indices
    """
    ps_B = powerset(A)  # powerset of A
    ps_B.pop(0)  # remove empty set
    m_A = np.zeros(Q.shape[0])
    for B in ps_B:
        dl = len(set(A) - set(B))
        m_A += ((-1) ** dl) * capacity(Q, B)
    return m_A
