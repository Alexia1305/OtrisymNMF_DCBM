import numpy as np
import time
import math
from scipy.sparse import find, csr_matrix
from .SVCA import SSPA
from .SVCA import SVCA
from .Utils import orthNNLS
from scipy.sparse import issparse
from scipy.sparse import diags


def OtrisymNMF_CD(X, r, numTrials=1, maxiter=1000, delta=1e-7, time_limit=300, init_method=None,
                  verbosity=1, init_seed=None):
    """
    Orthogonal Symmetric Nonnegative Matrix Trifactorization using Coordinate Descent.
    Given a symmetric matrix X >= 0, finds matrices W >= 0 and S >= 0 such that X ≈ WSW' with W'W=I.
    W is represented by:
    - v: indices of the nonzero columns of W for each row.
    - w: values of the nonzero elements in each row of W.

    Application to community detection:
        - X is the adjacency matrix of an undirected graph.
        - OtrisymNMF detects r communities.
        - v assigns each node to a community.
        - w indicates the importance of a node within its community.
        - S describes interactions between the r communities.

    Parameters:
        X : crs_matrix , shape (n, n)
            Symmetric nonnegative matrix (Adjacency matrix of an undirected graph).
        r : int
            Number of columns of W (Number of communities).
        numTrials : int, default=1
            Number of trials with different initializations.
        maxiter : int, default=1000
            Maximum iterations for each trial.
        delta : float, default=1e-7
            Convergence tolerance.
        time_limit : int, default=300
            Time limit in seconds.
        init_method : str, default=None
            Initialization method ("random", "SSPA", "SVCA", "SPA").
        verbosity : int, default=1
            Verbosity level (1 for messages, 0 for silent mode).
        init_seed : float, optional (default=None)
            Random seed for the initialization for the experiments

    Returns:
        w_best : np.array, shape (n,)
            Values of the nonzero elements for each row.
        v_best : np.array, shape (n,)
            Indices of the nonzero columns of W.
        S_best : np.array, shape (r, r)
            Central matrix.
        error_best : float
            Relative error ||X - WSW'||_F / ||X||_F.
        time_per_iteration : list[list[float]]
            Time per iteration for each trial.
            Each inner list contains the times (in seconds) for every iteration
            of that specific trial.
    """
    start_time = time.time()
    if not issparse(X):
        X = csr_matrix(X)
    n = X.shape[0]
    error_best = float('inf')
    time_per_iteration = []

    # Precomputations

    normX = np.sqrt(np.sum(X.data ** 2))
    I, J, V = find(X)
    perm = np.argsort(I)
    I = I[perm]
    J = J[perm]
    V = V[perm]
    rowStart = np.concatenate((
        [0],
        np.where(np.diff(I) != 0)[0] + 1,
        [len(I)]
    ))
    diagX = X.diagonal()

    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    for trial in range(numTrials):
        time_per_iteration.append([])
        if init_method is None:
            init_algo = "SSPA" if trial == 0 else "SVCA"
        else:
            init_algo = init_method

        # Initialization
        if init_algo == "random":
            if init_seed is not None:
                init_seed += 10 * trial
                np.random.seed(init_seed)
            base = np.arange(r)
            rest = np.random.randint(0, r, size=n - r)
            v = np.concatenate([base, rest])
            np.random.shuffle(v)
            w = np.random.rand(n)


        else:
            if init_seed is not None:
                init_seed += 10 * trial
            W = initialize_W(X, r, method=init_algo, init_seed=init_seed)
            w, v = extract_w_v(W, r)

        # Normalization of w
        nw = np.bincount(v, weights=w ** 2, minlength=r)
        nw = np.sqrt(nw)
        w = np.divide(w, nw[v], out=np.zeros_like(w), where=nw[v] != 0)

        # Compute of S
        prodVal = w[I] * w[J] * V  # w_i * w_j * X(i,j)
        S = np.zeros((r, r))
        np.add.at(S, (v[I], v[J]), prodVal)
        dgS = np.diag(S).copy()

        # Iterative update
        prev_error = compute_error(normX, S)
        error = prev_error



        for iteration in range(maxiter):
            start_it = time.time()
            if time.time() - start_time > time_limit:
                print('Time limit passed')
                break

            # Update W

            # Pre-calculations to avoid a double loop on ‘n’
            wp2 = np.zeros(r)
            S2 = S ** 2
            w2 = w ** 2
            for k in range(r):
                wp2[k] = np.sum(w2 * S2[v, k])
            # for each node
            for i in range(n):
                if time.time() - start_time > time_limit:
                    print('Time limit passed')
                    break
                # b coefficients for the r problems ax^4+bx^2+cx
                b = 2 * (wp2 - (w[i] * S[v[i], :]) ** 2) - 2 * diagX[i] * dgS

                # c coefficients
                ind = np.arange(rowStart[i], rowStart[i + 1])
                mask = J[ind] != i
                cols_i = J[ind[mask]]
                xip = V[ind[mask]]
                c = -4 * np.dot(xip * w[cols_i], S[v[cols_i], :])

                vi_new, wi_new, f_new = -1, -1, np.inf
                for k in range(r): # Test each community

                    # Cardan resolution for min ax^4+bx^2+cx
                    roots = cardan_depressed(4 * S2[k, k], 2 * b[k], c[k])

                    # best positive solution
                    x = np.sqrt(r / n)  # default value
                    min_value = S2[k, k] * (x ** 4) + b[k] * (x ** 2) + c[k] * x
                    for sol in roots:
                        value = S2[k, k] * (sol ** 4) + b[k] * (sol ** 2) + c[k] * sol
                        if sol > 0 and value < min_value:
                            x, min_value = sol, value

                    if S2[k, k] * x ** 4 + b[k] * x ** 2 + c[k] * x < f_new:
                        f_new, wi_new, vi_new = S2[k, k] * x ** 4 + b[k] * x ** 2 + c[k] * x, x, k

                # update wp2

                wp2 = wp2 - (w[i] * S[v[i], :]) ** 2 + (wi_new * S[vi_new, :]) ** 2

                # update of w et v for node i
                w[i], v[i] = wi_new, vi_new

            # Normalization of w
            nw = np.bincount(v, weights=w ** 2, minlength=r)
            nw = np.sqrt(nw)
            w = np.divide(w, nw[v], out=np.zeros_like(w), where=nw[v] != 0)

            # Compute of S
            prodVal = w[I] * w[J] * V  # w_i * w_j * X(i,j)
            S = np.zeros((r, r))
            np.add.at(S, (v[I], v[J]), prodVal)
            dgS = np.diag(S).copy()

            prev_error = error
            error = compute_error(normX, S)

            time_per_iteration[-1].append(round(time.time() - start_it, 4))

            if error < delta or abs(prev_error - error) < delta:
                break

        if iteration == maxiter - 1:
            print('Not converged')

        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error
        if verbosity > 0:
            print(
                f'Trial {trial + 1}/{numTrials} with {init_algo} in {iteration} iterations: Error {error:.4e} | Best: {error_best:.4e}')

        if error_best <= delta or time.time() - start_time > time_limit:
            break

    return w_best, v_best, S_best, error_best, time_per_iteration


def initialize_W(X, r, method="SVCA", init_seed=None):
    """ Initializes W based on the chosen method."""

    if method == "SSPA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):
            # Estimation of WS=W*S with SSPA
            # Attention the rank of X must be greater than r
            WS, K = SSPA(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Normalization of the columns of X
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)
            Xn = X @ D
            # Solve ||X-WSHO||_F with HOHO^T = D
            HO = orthNNLS(X, WS, Xn)
            # Transposition of the result
            W = HO.T
        else:
            # Estimation of W with SSPA
            # Attention the rank of X must be greater than r
            WS, K = SSPA(X, r, p, options=options)

            # Normalization of the X columns
            norm2x = np.sqrt(np.sum(X ** 2, axis=0))
            Xn = X * (1 / (norm2x + 1e-16))
            # Solve ||X-WSHO||_F with HOHO^T = D
            HO = orthNNLS(X, WS, Xn)
            # Transposition
            W = HO.T

    if method == "SVCA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):
            # Estimation of WS=W*S with SVCA
            # Attention the rank of X must be greater than r
            WS, K = SVCA(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Normalization of the X columns
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)
            Xn = X @ D
            # Solve ||X-WSHO||_F with HOHO^T = D
            HO = orthNNLS(X, WS, Xn)
            # Transposition
            W = HO.T


        else:
            # Estimation of WS=W*S with SVCA
            # Attention the rank of X must be greater than r
            WS, K = SVCA(X, r, p, options=options)

            # Normalization of the X columns
            norm2x = np.sqrt(np.sum(X ** 2, axis=0))
            Xn = X * (1 / (norm2x + 1e-16))
            # Solve ||X-WSHO||_F with HOHO^T = D
            HO = orthNNLS(X, WS, Xn)
            # Transposition
            W = HO.T

    return W


def extract_w_v(W, r):
    """ Extracts w and v from W."""
    w = np.max(W, axis=1)
    v = np.argmax(W, axis=1)
    # assign random communities if no assignement
    zero_indices = np.where(w == 0)[0]
    random_values = np.random.randint(0, r, size=zero_indices.shape[0])
    v[zero_indices] = random_values
    return w, v


def cardan_depressed(a, c, d, tol=1e-12):
    """ Cardano formula to find the roots of ax^3+cx+d=0 """
    if abs(a) < tol:
        if abs(c) < tol:
            return []
        else:
            return [-d / c]

    # b=0 t^3+pt+q
    p = c / a
    q = d / a
    Delta = 4 * (p ** 3) + 27 * (q ** 2)

    if abs(Delta) < tol:
        return [0]
    elif Delta > 0:  # one real solution
        sqrtD = np.sqrt(Delta / 27)
        return [np.cbrt((-q + sqrtD) / 2) + np.cbrt((-q - sqrtD) / 2)]

    else:  # 3 real different solutions or multiple solution

        r = 2 * np.sqrt(-p / 3)
        cos_arg = -q / 2 * np.sqrt(-27 / (p ** 3))
        cos_arg = np.clip(cos_arg, -1, 1)
        theta = np.arccos(cos_arg) / 3
        return r * np.cos(np.array([theta, theta + 2 * np.pi / 3, theta + 4 * np.pi / 3]))


def compute_error(normX, S):
    """ Computes error ||X - WSW'||_F / ||X||_F."""
    error = np.sqrt(normX ** 2 - np.linalg.norm(S, 'fro') ** 2) / normX
    return error


def Community_detection_SVCA(X, r, numTrials=1, verbosity=1, init_seed=None):
    """
        Perform community detection using the SVCA (Smooth VCA).

        Parameters:
        - X: ndarray or sparse matrix csr_matrix (Adjacency matrix of the graph)
        - r: int (Number of communities)
        - numTrials: int (Number of trials to find the best decomposition, default=1)
        - verbosity: int (Level of verbosity for printing progress, default=1)

        Returns:
        - w_best: ndarray (Importance of each node in its community)
        - v_best: ndarray (Community index for each node)
        - S_best: ndarray (Interaction matrix between communities)
        - error_best: float (Reconstruction error of the best trial)
        """

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
        W = initialize_W(X, r, method="SVCA", init_seed=init_seed)
        w, v = extract_w_v(W, r)
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

