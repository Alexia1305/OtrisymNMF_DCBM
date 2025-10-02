import numpy as np
import time
import math
from scipy.sparse import find, csr_matrix

from .SSPA import SSPA
from .SSPA import SVCA
from .Utils import orthNNLS
from scipy.sparse import issparse
from scipy.sparse import diags
from scipy.sparse.linalg import norm
from scipy.sparse import diags

def OtrisymNMF_CD(X, r, numTrials=1,update_rule="original", maxiter=1000, delta=1e-7, time_limit=300, init_method=None, verbosity=1,init_seed=None):
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

    "Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."
    2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2024.

    Parameters:
        X : crs_matrix , shape (n, n)
            Symmetric nonnegative matrix (Adjacency matrix of an undirected graph).
        r : int
            Number of columns of W (Number of communities).
        numTrials : int, default=1
            Number of trials with different initializations.
        update_rule : str, default="original"
            Update of W. Version original ("original") or with S directly updated ("S_direct")
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
                init_seed += 10*trial
            W = initialize_W(X, r,method=init_algo,init_seed=init_seed)
            w, v = extract_w_v(W,r)


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
        if update_rule=="S_direct":
            ################## S direct version ###################################
            # Matrix G = W^TXW  and  d = ||W(:,k)||^2
            d = np.ones(r)
            G = S.copy()
            for iteration in range(maxiter):
                start_it = time.time()
                if time.time() - start_time > time_limit:
                    print('Time limit passed')
                    break

                # Pré-calculs pour éviter une double boucle sur "n"
                wp2 = np.zeros(r)
                S2 = S ** 2
                w2 = w ** 2
                for k in range(r):
                    wp2[k] = np.sum(w2 * S2[v, k])

                for i in range(n):
                    if time.time() - start_time > time_limit:
                        print('Time limit passed')
                        break
                    # b coefficients for the r problems ax^4+bx^2+cx
                    if d[v[i]] != 0:
                        tempB = (w[i] / d[v[i]]) * safe_div_where_nonzero(G[v[i], :].flatten(), d)  # %w(i)*S(v(i),:)
                    else:
                        tempB = np.zeros(r)
                    b = 2 * (wp2 - tempB ** 2) - 2 * diagX[i] * dgS

                    # c coefficients
                    ind = np.arange(rowStart[i], rowStart[i + 1])
                    mask = J[ind] != i
                    cols_i = J[ind[mask]]
                    xip = V[ind[mask]]

                    tempC = safe_div_where_nonzero((xip * w[cols_i]), d[v[cols_i]])
                    d_safe = np.where(d == 0, 1, d)  # remplacer 0 par 1 temporairement pour éviter division par zéro
                    Gscl = G[v[cols_i], :] / d_safe[np.newaxis, :]
                    Gscl[:, d == 0] = 0
                    c = -4 * (tempC.T @ Gscl)

                    vi_new, wi_new, f_new = -1, -1, np.inf
                    for k in range(r):
                        if d[k] != 0:
                            S2kk = (G[k, k] / (d[k] ** 2)) ** 2
                        else:
                            S2kk = 0
                        # Cardan resolution for min ax^4+bx^2+cx
                        roots = cardan(4 * S2kk, 0, 2 * b[k], c[k])

                        # best positive solution
                        x = np.sqrt(r / n)  # default value
                        min_value = S2kk * (x ** 4) + b[k] * (x ** 2) + c[k] * x
                        for sol in roots:
                            value = S2kk * (sol ** 4) + b[k] * (sol ** 2) + c[k] * sol
                            if sol > 0 and value < min_value:
                                x, min_value = sol, value

                        if S2kk * x ** 4 + b[k] * x ** 2 + c[k] * x < f_new:
                            f_new, wi_new, vi_new = S2kk * x ** 4 + b[k] * x ** 2 + c[k] * x, x, k

                    # for the update of p
                    G_old = G[v[i], :].flatten()
                    G_best = G[vi_new, :].flatten()
                    oldRow_p = safe_div_where_nonzero((G_old ** 2), (d[v[i]] * (d ** 2)))
                    bestRow_p = safe_div_where_nonzero((G_best ** 2), (d[vi_new] * (d ** 2)))

                    # update of d
                    d[v[i]] -= w[i] ** 2
                    d[vi_new] += wi_new ** 2

                    # update of G
                    coeffVec = xip.flatten() * w[cols_i]

                    # Remove the old contribution
                    delta_old = np.zeros(r)
                    np.add.at(delta_old, v[cols_i], w[i] * coeffVec)
                    delta_old[v[i]] *= 2
                    G[v[i], :] -= delta_old
                    G[:, v[i]] = G[v[i], :].T

                    # Add the new contribution
                    delta_new = np.zeros(r)
                    np.add.at(delta_new, v[cols_i], wi_new * coeffVec)
                    delta_new[vi_new] *= 2
                    G[vi_new, :] += delta_new
                    G[:, vi_new] = G[vi_new, :].T

                    # Update of diagonal
                    G[v[i], v[i]] -= w[i] ** 2 * diagX[i]
                    G[vi_new, vi_new] += wi_new ** 2 * diagX[i]

                    # update of  wp2

                    newRow_old = safe_div_where_nonzero((G[v[i], :].flatten() ** 2), (d[v[i]] * (d ** 2)))
                    newRow_best = safe_div_where_nonzero((G[vi_new, :].flatten() ** 2), (d[vi_new] * (d ** 2)))

                    if v[i] != vi_new:
                        wp2 = wp2 - oldRow_p - bestRow_p + newRow_old + newRow_best
                    else:
                        wp2 = wp2 - oldRow_p + newRow_old

                    tmp = safe_div_where_nonzero((G[:, v[i]].flatten()) ** 2, d)
                    if d[v[i]] != 0:
                        wp2[v[i]] = np.sum(tmp) / (d[v[i]] ** 2)
                        dgS[v[i]] = G[v[i], v[i]] / (d[v[i]] ** 2)
                    else:
                        wp2[v[i]] = 0
                        dgS[v[i]] = 0

                    tmp = safe_div_where_nonzero(G[:, vi_new] ** 2, d)
                    if d[vi_new] != 0:
                        dgS[vi_new] = G[vi_new, vi_new] / (d[vi_new] ** 2)
                        wp2[vi_new] = np.sum(tmp) / (d[vi_new] ** 2)
                    else:
                        wp2[vi_new] = 0
                        dgS[vi_new] = 0
                        # Mise à jour de dgS

                    # Update of w et v with the new values
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

                # Matrix G=WtXW and d=||W(:,k)||^2
                d = np.ones(r)
                G = S.copy()

                prev_error = error
                error = compute_error(normX, S)
                time_per_iteration[-1].append(time.time()-start_it)
                if error < delta or abs(prev_error - error) < delta:
                    break
        else:
            ################### Original version #################################
            for iteration in range(maxiter):
                start_it = time.time()
                if time.time() - start_time > time_limit:
                    print('Time limit passed')
                    break

                # Update W

                # Pré-calculs pour éviter une double boucle sur "n"
                wp2 = np.zeros(r)
                S2 = S**2
                w2 = w**2
                for k in range(r):
                    wp2[k] = np.sum(w2 * S2[v, k])


                for i in range(n):
                    if time.time() - start_time > time_limit:
                        print('Time limit passed')
                        break
                    # b coefficients for the r problems ax^4+bx^2+cx
                    b = 2 * (wp2 - (w[i] * S[v[i], :]) ** 2) - 2 * diagX[i]*dgS

                    # c coefficients
                    ind = np.arange(rowStart[i], rowStart[i + 1])
                    mask = J[ind] != i
                    cols_i = J[ind[mask]]
                    xip = V[ind[mask]]
                    c = -4 * np.dot(xip * w[cols_i], S[v[cols_i], :])

                    vi_new, wi_new, f_new = -1, -1, np.inf
                    for k in range(r):

                        # Cardan resolution for min ax^4+bx^2+cx
                        roots = cardan(4 * S2[k,k], 0, 2 * b[k], c[k])

                        # best positive solution
                        x = np.sqrt(r / n) # default value
                        min_value = S2[k,k] * (x ** 4) + b[k] * (x ** 2) + c[k] * x
                        for sol in roots:
                            value = S2[k,k] * (sol ** 4) + b[k] * (sol ** 2) + c[k] * sol
                            if sol > 0 and value < min_value:
                                x, min_value = sol, value

                        if S2[k,k] * x ** 4 + b[k] * x ** 2 + c[k] * x < f_new:
                            f_new, wi_new, vi_new = S2[k, k] * x ** 4 + b[k] * x ** 2 + c[k] * x, x, k

                    # update wp2

                    wp2 = wp2 - (w[i] * S[v[i], :]) ** 2 + (wi_new * S[vi_new, :]) ** 2

                    # Mise à jour des valeurs de w et v
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

        if iteration == maxiter-1:
            print('Not converged')

        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error
        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with {init_algo} in {iteration} iterations: Error {error:.4e} | Best: {error_best:.4e}')

        if error_best <= delta or time.time() - start_time > time_limit:
            break


    return w_best, v_best, S_best, error_best,time_per_iteration


def initialize_W(X, r, method="SVCA",init_seed=None):
    """ Initializes W based on the chosen method."""

    if method == "SSPA":
        n = X.shape[0]
        p=max(2,math.floor(0.1*n/r))

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):

            WO, K = SSPA(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Créer matrice diagonale inverses des normes
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)  # matrice diagonale sparse (n x n)

            # Normaliser X par colonnes : multiplication à droite
            Xn = X @ D

            HO = orthNNLS(X, WO, Xn)

            # Transposition du résultat
            W = HO.T
        else :
            WO, K = SSPA(X, r, p,options=options )

            norm2x = np.sqrt(np.sum(X ** 2, axis=0))  # Calcul de la norme L2 sur chaque colonne de X
            Xn = X * (1 / (norm2x + 1e-16))  # Normalisation de X (évite la division par zéro)


            HO = orthNNLS(X, WO, Xn)

            # Transposition du résultat
            W = HO.T

    if method == "SVCA":
        n = X.shape[0]
        p = max(2, math.floor(0.1 * n / r))

        options = {'average': 1}
        if init_seed is not None:
            np.random.seed(init_seed)

        if issparse(X):

            WO, K = SVCA(X.tocsc(), r, p, options=options)
            norm2x_squared = X.multiply(X).sum(axis=0)  # matrice 1 x n (sparse)
            norm2x_squared = np.array(norm2x_squared).ravel()
            norm2x = np.sqrt(norm2x_squared)
            norm2x_safe = norm2x + 1e-16

            # Créer matrice diagonale inverses des normes
            inv_norms = 1.0 / norm2x_safe
            D = diags(inv_norms)  # matrice diagonale sparse (n x n)

            # Normaliser X par colonnes : multiplication à droite
            Xn = X @ D

            HO = orthNNLS(X, WO, Xn)

            # Transposition du résultat
            W = HO.T


        else:
            WO, K = SVCA(X, r, p, options=options)

            norm2x = np.sqrt(np.sum(X ** 2, axis=0))  # Calcul de la norme L2 sur chaque colonne de X
            Xn = X * (1 / (norm2x + 1e-16))  # Normalisation de X (évite la division par zéro)

            HO = orthNNLS(X, WO, Xn)

            # Transposition du résultat
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






def cardan(a, b, c, d,tol=1e-12):
    """ Cardano formula to solve ax^3+bx^2+cx+d=0"""
    if abs(a) < tol:
        if abs(b) < tol:
            if abs(c) < tol:
                roots = []
                return roots
            else:
                root1 = -d / c
                roots = [root1]
                return roots

        delta = c ** 2 - 4 * b * d
        root1 = (-c + math.sqrt(delta)) / (2 * b)
        root2 = (-c - math.sqrt(delta)) / (2 * b)

        if root1 == root2:
            roots = [root1]
        else:
            roots = [root1, root2]
        return roots

    p = -(b ** 2 / (3 * a ** 2)) + c / a
    q = ((2 * b ** 3) / (27 * a ** 3)) - ((9 * c * b) / (27 * a ** 2)) + (d / a)
    delta = -(4 * p ** 3 + 27 * q ** 2)



    if abs(delta) < tol:
        if abs(p) < tol and abs(q) < tol:
            root1 = 0
            roots = [root1]
        else:
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            roots = [root1, root2]
        return roots
    elif delta < -tol:
        u = (-q + math.sqrt(-delta / 27)) / 2
        v = (-q - math.sqrt(-delta / 27)) / 2

        if abs(u) < tol:
            u = 0
        elif u < 0:
            u = -(-u) ** (1 / 3)
        else :
            u = u ** (1 / 3)

        if abs(v)< tol:
            v = 0
        elif v < 0:
            v = -(-v) ** (1 / 3)
        else :#v > 0:
            v = v ** (1 / 3)


        root1 = u + v - (b / (3 * a))
        roots = [root1]
        return roots
    else:
        epsilon = -1e-30
        phi = math.acos(-(q / 2) * math.sqrt(-27 / (p ** 3 + epsilon)))
        z1 = 2 * math.sqrt(-p / 3) * math.cos(phi / 3)
        z2 = 2 * math.sqrt(-p / 3) * math.cos((phi + 2 * math.pi) / 3)
        z3 = 2 * math.sqrt(-p / 3) * math.cos((phi + 4 * math.pi) / 3)

        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))

        roots = [root1, root2, root3]
        return roots

def compute_error(normX, S):
    """ Computes error ||X - WSW'||_F / ||X||_F."""
    error = np.sqrt(normX**2-np.linalg.norm(S, 'fro')**2)/normX
    return error

def Community_detection_SVCA(X, r, numTrials=1,verbosity=1,init_seed=None):
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

        # Placeholder for proper initialization functions
        if init_seed is not None:
            init_seed += 10 * trial
        W = initialize_W(X, r, method="SVCA",init_seed=init_seed)
        w, v = extract_w_v(W,r)
        # Normalization of w
        nw = np.bincount(v, weights=w ** 2, minlength=r)
        nw = np.sqrt(nw)
        w = np.divide(w, nw[v], out=np.zeros_like(w), where=nw[v] != 0)

        # Compute of S
        prodVal = w[I] * w[J] * V  # w_i * w_j * X(i,j)
        S = np.zeros((r, r))
        np.add.at(S, (v[I], v[J]), prodVal)
        dgS = np.diag(S).copy()


        error = compute_error(normX, S)




        if error <= error_best:
            w_best, v_best, S_best, error_best = w, v, S, error

        if verbosity > 0:
            print(f'Trial {trial + 1}/{numTrials} with SVCA: Error {error:.4e} | Best: {error_best:.4e}')

    return w_best, v_best, S_best, error_best

def safe_div_where_nonzero(A, B):
    """
    Fait A / B là où B != 0, sinon met 0. Évite les warnings NumPy.
    A et B doivent avoir la même forme.
    """
    result = np.zeros_like(A, dtype=float)
    mask = B != 0
    result[mask] = A[mask] / B[mask]
    return result
