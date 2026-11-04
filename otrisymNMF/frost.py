import numpy as np
import math
import time
from scipy.sparse import find, csr_array
from scipy.sparse import issparse
from .svca_initialization import initialize_Z
from .utils import compute_error,regularized_normalized_A


def frost(X, r, numTrials=1, maxiter=1000, delta=1e-5, time_limit=1000000, init_method=None,
                  verbosity=1, init_seed=None, regularized_normalized=True,init_partition=None ):
    """
    Heuristic for the Orthogonal Symmetric Nonnegative Matrix Trifactorization problem:
    Given a symmetric matrix X >= 0, finds matrices Z >= 0 and S >= 0 such that X ≈ ZSZ' with Z'Z=I.
    The factorization is computed with respect to the Frobenius norm, i.e., it aims to solve:

       min_{Z >= 0, S >= 0} || X - Z S Z^T ||_F^2
       subject to Z^T Z = I

    Z is represented by:
    - v: indices of the nonzero columns of Z for each row.
    - w: values of the nonzero elements in each row of Z.

    Application to community detection:
        - X is the adjacency matrix of an undirected graph.
        - r is the number of communities.
        - v assigns each node to a community (v[i] is the community index of node i).
        - w indicates the importance of each node within its community.
        - S describes interactions between the r communities.

    Parameters:
        X : crs_array , shape (n, n)
            Symmetric nonnegative matrix (Adjacency matrix of an undirected graph).
        r : int
            Number of columns of Z (Number of communities).
        numTrials : int, default=1
            Number of trials with different initializations.
        maxiter : int, default=1000
            Maximum iterations for each trial.
        delta : float, default=1e-7
            Convergence tolerance (Stop if error<delta or error_prec-error<delta).
        time_limit : int, default=300
            Time limit in seconds.
        init_method : str, default=SVCA
            Initialization method ("random", "SVCA", "SSPA", "SPA").
        verbosity : int, default=1
            (1 for messages, 0 for silent mode).
        init_seed : float, optional (default=None)
            Random seed for the initialization for the experiments
        regularized_normalized: boolean, default=True 
            Use the regularized normalized adjacency matrix 

    Returns:
        w_best : np.array, shape (n,)
            Values of the nonzero elements for each row.
        v_best : np.array, shape (n,)
            Indices of the nonzero columns of Z.
        S_best : np.array, shape (r, r)
            Central matrix.
        error_best : float
            Relative error ||X - ZSZ'||_F / ||X||_F.
        time_per_iteration : list[list[float]]
            Time per iteration for each trial.
            Each inner list contains the times (in seconds) for every iteration
            of that specific trial.
    """
    start_time = time.time()
    if not issparse(X):
        X = csr_array(X)
    n = X.shape[0]
    error_best = float('inf')
    time_per_iteration = []
    # Verification that there is no zero row in X (isolated node with no information)
    if np.any(X.indptr[:-1] == X.indptr[1:]):
        raise ValueError(
            "The matrix X contains at least one zero row. Please remove empty rows (nodes without connections) during preprocessing.")
    if regularized_normalized:
        X=regularized_normalized_A(X)

        
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
    has_diagonal = np.any(diagX != 0)
    rows = []
    for i in range(n):
        ind = np.arange(rowStart[i], rowStart[i + 1])
        mask = J[ind] != i
        rows.append((J[ind[mask]], V[ind[mask]]))
    
    if verbosity > 0:
        print(f'Running {numTrials} Trials in Series')

    for trial in range(numTrials):
        time_per_iteration.append([])
        if init_partition is not None :
            init_algo ="init_partition"
            v = init_partition
            w = np.asarray(X.sum(axis=1)).ravel().astype(float)
            
        else: 
            if init_method is None:
                init_algo = "SVCA"
            else:
                init_algo = init_method

            # INITIALIZATION
            if init_algo == "random":
                if init_seed is not None:
                    init_seed += 10 * trial
                    np.random.seed(init_seed)
                base = np.arange(r)
                rest = np.random.randint(0, r, size=n - r)
                v = np.concatenate([base, rest])
                np.random.shuffle(v)
                w = np.random.rand(n)
            # SVCA SSPA SPA
            else:
                if init_seed is not None:
                    init_seed += 10 * trial
                w, v = initialize_Z(X, r, method=init_algo, init_seed=init_seed)


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

            # UPDATE Z
            # Pre-calculations to avoid a double loop on ‘n’
            wp2 = np.zeros(r)
            S2 = S ** 2
            dgS2 = np.diag(S2)
            cardan_a = list(4 * dgS2)
            dgS2 = list(dgS2)
            S_rows = list(S)
            w2 = w ** 2
            for k in range(r):
                wp2[k] = np.sum(w2 * S2[v, k])
            # Update of each row (node)
            perm = np.random.permutation(n)
            for i in perm:
                if time.time() - start_time > time_limit:
                    print('Time limit passed')
                    break
                # b coefficients for the r problems ax^4+bx^2+cx
                contribution = (w[i] * S_rows[v[i]]) ** 2
                b = 2 * (wp2 - contribution)
                if has_diagonal:
                    b -= 2 * diagX[i] * dgS
                # c coefficients
                cols_i, xip = rows[i]
                c = -4 * np.dot(xip * w[cols_i], S[v[cols_i], :])
                vi_new, wi_new, f_new = -1, -1, np.inf
                for k, (a_k, b_k, c_k, cardan_ak) in enumerate(zip(dgS2, b, c, cardan_a)):  # Test each community
                    # Cardan resolution for min ax^4+bx^2+cx
                    roots = cardan_depressed(cardan_ak, 2 * b_k, c_k)
                    # best positive solution
                    x = 0
                    min_value = 0
                    for sol in roots:
                        if sol > 0:
                            value = a_k * (sol ** 4) + b_k * (sol ** 2) + c_k * sol
                            if value < min_value:
                                x, min_value = sol, value
                    if min_value < f_new:
                        f_new, wi_new, vi_new = min_value, x, k
                # update wp2
                wp2 -= contribution
                wp2 += (wi_new * S_rows[vi_new]) ** 2
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
        sqrtD = math.sqrt(Delta / 27)
        return [math.cbrt((-q + sqrtD) / 2) + math.cbrt((-q - sqrtD) / 2)]

    else:  # 3 real different solutions or multiple solution

        r = 2 * math.sqrt(-p / 3)
        cos_arg = -q / 2 * math.sqrt(-27 / (p ** 3))
        cos_arg = max(-1.0, min(1.0, cos_arg))
        theta = math.arccos(cos_arg) / 3
        return [ r * math.cos(theta), r * math.cos(theta + 2 * math.pi / 3), r * math.cos(theta + 4 * math.pi / 3)]
