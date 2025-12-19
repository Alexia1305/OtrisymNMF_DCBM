
import numpy as np

def compute_error(normX, S):
    """ Computes error ||X - ZSZ'||_F / ||X||_F."""
    error = np.sqrt(normX ** 2 - np.linalg.norm(S, 'fro') ** 2) / normX
    return error