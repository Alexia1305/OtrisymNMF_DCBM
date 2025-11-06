import numpy as np
from scipy.sparse import issparse, diags
from sklearn.cluster import KMeans
from .SVCA import SSPA
from .SVCA import SVCA

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


def alternatingONMF(X, r, maxiter=100, delta=1e-6, init_algo="k_means"):
    """
        Solve Orthogonal Nonnegative Matrix Factorization (ONMF) using the
        two-block coordinate descent (2-BCD) method.

        The problem is formulated as:

            min_{W, H} || X - W H ||_F^2
            subject to H >= 0 and H H^T = I_r

        The algorithm alternates between updating H and W using closed-form
        solutions.

        Notes
        -----
        - The columns of W correspond to centroids of a subset of the columns of X.
          Therefore, W is nonnegative if X is nonnegative.
        - The algorithm can also be applied to input matrices X that contain
          negative entries, in which case W may not be nonnegative.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data matrix to be factorized.

        r : int
            Factorization rank.

        maxiter : int, optional, default=100
            Maximum number of iterations.

        delta : float, optional, default=1e-6
            Convergence tolerance. The algorithm stops when the relative error does
            not change by more than `delta` between two consecutive iterations.

        init_algo : {"k_means", "SVCA"}, optional, default="k_means"
            Initialization method for W and H:
            - "k_means" : initialize using k-means clustering on normalized columns of X.
            - "SVCA"    : initialize using the SVCA method.

        Returns
        -------
        W : ndarray of shape (m, r)

        H : ndarray of shape (r, n)
            Coefficient matrix, with H >= 0 and H H^T = I_r.
        err : float
            Final relative error defined as ||X - W H||_F / ||X||_F.

        References
        ----------
        F. Pompili, N. Gillis, P.-A. Absil, and F. Glineur,
        "Two Algorithms for Orthogonal Nonnegative Matrix Factorization with
        Application to Clustering", Neurocomputing, 141:15–25, 2014.
        """
    if init_algo == "k_means":
        # Initialization kmeans
        n = X.shape[1]
        Xnorm = np.zeros_like(X, dtype=float)

        # Normalization of the X columns
        for i in range(n):
            col_norm = np.linalg.norm(X[:, i], 2)
            if col_norm != 0:
                Xnorm[:, i] = X[:, i] / col_norm
            else:
                Xnorm[:, i] = X[:, i]

        kmeans = KMeans(n_clusters=r, n_init=10, max_iter=1000).fit(Xnorm.T)
        a = kmeans.labels_

        H = np.zeros((r, n))
        for i in range(n):
            H[a[i], i] = 1.0

        # Orthogonalization of the rows
        for k in range(r):
            nw = np.linalg.norm(H[k, :], 2)
            if nw != 0:
                H[k, :] /= nw

        W = X @ H.T

    elif init_algo == "SVCA":
        n = X.shape[1]
        p = max(2, int(np.floor(0.1 * n / r)))
        options = {"average": 1}
        W, K = SVCA(X, r, p, options)

    m, n = X.shape
    m, r = W.shape

    norm2x = np.sqrt(np.sum(X ** 2, axis=0))
    Xn = X / (norm2x + 1e-16)
    normX2 = np.sum(X ** 2)

    k = 1
    e = []
    H = np.zeros((r, n))

    while k <= maxiter and (k <= 3 or abs(e[-1] - e[-2]) > delta if len(e) > 2 else True):
        H = orthNNLS(X, W, Xn)

        norm2h = np.sqrt(np.sum(H.T ** 2, axis=0)) + 1e-16
        H = (1.0 / norm2h).reshape(-1, 1) * H

        W = X @ H.T

        err = (normX2 - np.sum(W ** 2)) / normX2
        if err < 0:
            err = 0
        else:
            err = np.sqrt(err)
        e.append(err)

        k += 1

    return W, H, e[-1]



