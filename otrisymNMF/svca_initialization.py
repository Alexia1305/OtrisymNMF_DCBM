
import numpy as np
import math
from scipy.sparse import find, csr_matrix
from .svca import sspa
from .svca import svca
from scipy.sparse import issparse
from scipy.sparse import diags
from .utils import compute_error

def community_detection_svca(X, r, numTrials=1, verbosity=1, init_seed=None):
    """
        Perform community detection using SVCA (Smooth VCA).
        Find a first approximation of  Z >= 0 and S >= 0 such that X ≈ ZSZ' with Z'Z=I.
        Z is represented by:
        - v: indices of the nonzero columns of Z for each row.
        - w: values of the nonzero elements in each row of Z.

        Parameters:
        - X: ndarray or sparse matrix csr_matrix (Adjacency matrix of a graph)
        - r: int (Number of communities)
        - numTrials: int (Number of trials to find the best decomposition, default=1)
        - verbosity: int (1 for messages, 0 for silent mode).
        - init_seed : float, optional (default=None)
                      Random seed for the initialization for the experiments

        Returns:
        - w_best: ndarray (Importance of each node in its community)
        - v_best: ndarray (Community index for each node)
        - S_best: ndarray (Interaction matrix between communities)
        - error_best: float (Reconstruction error of the best trial)
        """
    # Verification that there is no zero row in X (isolated node with no information)
    if any(X.indptr[i] == X.indptr[i + 1] for i in range(X.shape[0])):
        raise ValueError(
            "The matrix X contains at least one zero row. Please remove empty rows (nodes without connections) during preprocessing.")

    I, J, V = find(X)
    perm = np.argsort(I)
    I = I[perm]
    J = J[perm]
    V = V[perm]
    error_best = float('inf')
    normX = np.sqrt(np.sum(X.data ** 2))
    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    for trial in range(numTrials):

        if init_seed is not None:
            init_seed += 10 * trial
        Z = initialize_Z(X, r, method="SVCA", init_seed=init_seed)
        w, v = extract_w_v(Z, r)
        # Normalization of w
        nw = np.bincount(v, weights=w ** 2, minlength=r)
        nw = np.sqrt(nw)
        w = np.divide(w, nw[v], out=np.zeros_like(w), where=nw[v] != 0)

        # Compute of S
        prodVal = w[I] * w[J] * V  # w_i * w_j * X(i,j)
        S = np.zeros((r, r))
        np.add.at(S, (v[I], v[J]), prodVal)

        error = compute_error(normX, S)

        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error

        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with SVCA: Error {error:.4e} | Best: {error_best:.4e}')

    return w_best, v_best, S_best, error_best

def initialize_Z(X, r, method="SVCA", init_seed=None):
    """
    Initialize Z via smooth separable NMF
    Find a first approximation of  Z >= 0 such that X ≈ ZSZ' with Z'Z=I.

        Z is represented by:
        - v: indices of the nonzero columns of Z for each row.
        - w: values of the nonzero elements in each row of Z.

    Parameters:
    - X: ndarray or sparse matrix csr_matrix (Adjacency matrix of the graph)
    - r: int (Number of communities)
    - method : str, default=SVCA
        method ("random", "SVCA", "SSPA", "SPA").
    - init_seed : float, optional (default=None)
        Random seed for the initialization for the experiments

    Returns:
    - w: ndarray (Importance of each node in its community)
    - v: ndarray (Community index for each node)

    """

    if method == "SSPA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))  # default value

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):
            # Estimation of ZS=Z*S with SSPA
            # Attention the rank of X must be greater than r
            ZS, K = sspa(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Normalization of the columns of X
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)
            Xn = X @ D
            # Solve ||X-ZSHO||_F with HOHO^T = D
            HO = orthNNLS(X, ZS, Xn)
            # Transposition of the result
            Z = HO.T
        else:
            # Estimation of Z with SSPA
            # Attention the rank of X must be greater than r
            ZS, K = sspa(X, r, p, options=options)

            # Normalization of the X columns
            norm2x = np.sqrt(np.sum(X ** 2, axis=0))
            Xn = X * (1 / (norm2x + 1e-16))
            # Solve ||X-ZSHO||_F with HOHO^T = D
            HO = orthNNLS(X, ZS, Xn)
            # Transposition
            Z = HO.T

    if method == "SVCA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):
            # Estimation of ZS=Z*S with SVCA
            # Attention the rank of X must be greater than r
            ZS, K = svca(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Normalization of the X columns
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)
            Xn = X @ D
            # Solve ||X-ZSHO||_F with HOHO^T = D
            HO = orthNNLS(X, ZS, Xn)
            # Transposition
            Z = HO.T


        else:
            # Estimation of ZS=Z*S with SVCA
            # Attention the rank of X must be greater than r
            ZS, K = svca(X, r, p, options=options)

            # Normalization of the X columns
            norm2x = np.sqrt(np.sum(X ** 2, axis=0))
            Xn = X * (1 / (norm2x + 1e-16))
            # Solve ||X-ZSHO||_F with HOHO^T = D
            HO = orthNNLS(X, ZS, Xn)
            # Transposition
            Z = HO.T
        w, v = extract_w_v(Z, r)

    return w, v


def extract_w_v(Z, r):
    """ Extracts w and v from Z.
        - v: indices of the nonzero columns of Z for each row.
        - w: values of the nonzero elements in each row of Z.
    """
    w = np.max(Z, axis=1)
    v = np.argmax(Z, axis=1)
    # assign random communities if no assignment
    zero_indices = np.where(w == 0)[0]
    random_values = np.random.randint(0, r, size=zero_indices.shape[0])
    v[zero_indices] = random_values
    return w, v

def orthNNLS(M, U, Mn=None):
    """
    Solves the following optimization problem:
    min_{norm2v >= 0, V >= 0 and VV^T = D} ||M - U * V||_F^2

    Parameters:
        M (numpy.ndarray or csr_matrix): Matrix M of size (m, n).
        U (numpy.ndarray ): Matrix U of size (m, r).
        Mn (numpy.ndarray or csr_matrix, optional): Normalized columns of M. If None, it will be computed.

    Returns:
        V (numpy.ndarray): The matrix V of size (r, n) that approximates M.
        norm2v (numpy.ndarray): The squared norms of the columns of V.
    see F. Pompili, N. Gillis, P.-A. Absil and F. Glineur, "Two Algorithms for
    Orthogonal Nonnegative Matrix Factorization with Application to
    Clustering", Neurocomputing 141, pp. 15-25, 2014.
    """

    if Mn is None:
        # Normalize columns of M
        if issparse(M):
            norm2x_squared = M.multiply(M).sum(axis=0)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)

            # Normalization of the columns of M
            Mn = M @ D
        else:
            norm2m = np.sqrt(np.sum(M ** 2, axis=0))
            Mn = M * (1 / (norm2m + 1e-16))  # Avoid division by zero

    m, n = Mn.shape
    m_, r = U.shape

    # Normalize columns of U
    norm2u = np.sqrt(np.sum(U ** 2, axis=0))  # norm2u is the L2 norm of each column of U
    Un = U * (1 / (norm2u + 1e-16))  # Avoid division by zero
    if issparse(M):
        Mn = Mn.tocsc()
        M = M.tocsc()

    # Calculate the matrix A, which is the angles between columns of M and U
    A = Mn.T @ Un  # A is (n, r), matrix of angles

    # Find the index of the maximum value in each row of A (best column of U to approximate each column of M)
    b = np.argmax(A, axis=1)  # Indices of the best matching column in U

    # Initialize V with zeros
    V = np.zeros((r, n))

    # Assign the optimal weights to V(b(i), i)
    for i in range(n):
        if issparse(M):
            V[b[i], i] = (M[:, i].T @ U[:, b[i]] )[0] / norm2u[b[i]] ** 2
        else:
            V[b[i], i] = np.dot(M[:, i].T, U[:, b[i]]) / norm2u[b[i]] ** 2

    return V

