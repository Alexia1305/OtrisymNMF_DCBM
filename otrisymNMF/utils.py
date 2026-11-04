
import numpy as np
from scipy.sparse import issparse

def compute_error(normX, S):
    """ Computes error ||X - ZSZ'||_F / ||X||_F."""
    error = np.sqrt(normX ** 2 - np.linalg.norm(S, 'fro') ** 2) / normX
    return error

def regularized_normalized_A(A, delta=1.0):
    """Compute (D + delta*d_mean*I)^(-1/2) A (D + delta*d_mean*I)^(-1/2)."""
    if delta < 0:
        raise ValueError("delta value ")

    if not issparse(A):
        A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A doit être carrée.")

    degrees = np.asarray(A.sum(axis=1)).ravel()
    d_mean = degrees.mean() if degrees.size else 0.0
    d_reg = degrees + delta * d_mean

    inv_sqrt = np.zeros_like(d_reg, dtype=float)
    positive = d_reg > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(d_reg[positive])

    if issparse(A):
        return (
            A.multiply(inv_sqrt[:, None])
            .multiply(inv_sqrt[None, :])
            .tocsr()
        )

    return inv_sqrt[:, None] * A * inv_sqrt[None, :]


