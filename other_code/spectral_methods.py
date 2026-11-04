import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def SCORE(X, r, init_seed=None):
    """
    SCORE (Jin, 2015) with ratio truncation at +/-log(n).

    Parameters
    ----------
    X : array-like or scipy.sparse matrix, shape (n, n)
        Symmetric, nonnegative adjacency matrix of a connected graph.
    r : int
        Number of communities, with 1 <= r < n.
    init_seed : int, optional
        Seed for the eigensolver and K-means.

    Returns
    -------
    np.ndarray, shape (n,)
        Community labels, numbered from 0 to r - 1.
    """
    X = csr_matrix(X, dtype=np.float64)
    X.eliminate_zeros()
    n = X.shape[0]

    if X.shape != (n, n):
        raise ValueError("X must be square.")
    if not isinstance(r, (int, np.integer)) or not 1 <= r < n:
        raise ValueError("r must be an integer satisfying 1 <= r < n.")
    if not np.all(np.isfinite(X.data)) or np.any(X.data < 0):
        raise ValueError("X must contain finite, nonnegative values.")
    if (X != X.T).nnz:
        raise ValueError("X must be symmetric.")

    if r == 1:
        return np.zeros(n, dtype=int)

    
    # Compute the r eigenvectors with largest eigenvalue magnitudes.
    rng = np.random.default_rng(init_seed)
    eigvals, eigvecs = eigsh(
        X, k=r, which="LM", v0=rng.standard_normal(n)
    )

    # Put the Perron eigenvector first, including for bipartite graphs.
    idx = np.argsort(-np.abs(eigvals))
    perron = np.argmax(eigvals)
    idx = np.concatenate(([perron], idx[idx != perron]))
    U = eigvecs[:, idx]

    # Orient the denominator eigenvector to be positive.
    eta1 = U[:, 0].copy()
    if eta1.sum() < 0:
        eta1 = -eta1

    

    # Entrywise ratios and truncation: Jin (2015), equations (2.2)-(2.3).
    R = U[:, 1:] / eta1[:, None]
    R = np.clip(R, -np.log(n), np.log(n))

    return KMeans(
        n_clusters=r,
        n_init=100,
        random_state=init_seed
    ).fit_predict(R)

def SCORE_plus(X, r, init_seed=None, delta=0.1, t=0.1,
               n_init=100):
    """
    SCORE+ (Jin, Ke and Luo).

    Parameters
    ----------
    X : ndarray or scipy sparse matrix
        Symmetric, nonnegative adjacency matrix.
    r : int
        Number of communities, 1 <= r < n.
    init_seed : int or None
        Random seed for eigsh and KMeans.
    delta : float
        Ridge coefficient multiplying maximum degree.
    t : float
        Threshold for including the (r+1)-th eigenvector.
    n_init : int
        Number of KMeans initializations.

    Returns
    -------
    labels : ndarray, shape (n,)
        Estimated community labels.
    """
    X = csr_matrix(X, dtype=np.float64, copy=True)
    X.sum_duplicates()
    X.eliminate_zeros()

    n, m = X.shape

    if n != m:
        raise ValueError("X must be square.")
    if not isinstance(r, (int, np.integer)) or not 1 <= r < n:
        raise ValueError("Require 1 <= r < n.")
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and positive.")
    if t <= 0 or not np.isfinite(t):
        raise ValueError("t must be finite and positive.")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer.")
    if not np.all(np.isfinite(X.data)) or np.any(X.data < 0):
        raise ValueError("X must be finite and nonnegative.")
    if (X != X.T).nnz != 0:
        raise ValueError("X must be symmetric.")

    degrees = np.asarray(X.sum(axis=1)).ravel()

    if np.any(degrees == 0):
        raise ValueError(
            "Isolated nodes detected. Remove them before SCORE+."
        )

    
    if r == 1:
        return np.zeros(n, dtype=int)

    # 1. Regularized normalized adjacency matrix
    scale = 1.0 / np.sqrt(
        degrees + delta * degrees.max()
    )
    S = diags(scale, format="csr")
    L = (S @ X @ S).tocsr()

    # 2. Leading r+1 eigenpairs by absolute magnitude
    rng = np.random.default_rng(init_seed)
    v0 = rng.standard_normal(n)

    if r + 1 < n:
        try:
            eigvals, eigvecs = eigsh(
                L,
                k=r + 1,
                which="LM",
                v0=v0,
                tol=1e-10
            )
        except ArpackNoConvergence as exc:
            raise RuntimeError(
                "SCORE+ eigensolver did not converge."
            ) from exc
    else:
        eigvals, eigvecs = np.linalg.eigh(L.toarray())

    # Order by decreasing eigenvalue magnitude
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    U = eigvecs[:, idx]

    # The Perron eigenvalue must be first.
    # For a connected nonnegative matrix, it is positive.
    perron = np.argmax(eigvals)

    if perron != 0:
        # Handle possible magnitude ties explicitly
        
        idx = np.concatenate((
            [perron],
            np.arange(len(eigvals))[
                np.arange(len(eigvals)) != perron
            ]
        ))
        eigvals = eigvals[idx]
        U = U[:, idx]

    # Orient Perron eigenvector positively
    if U[:, 0].sum() < 0:
        U[:, 0] *= -1

    

    # 3. Weak-signal criterion
    

    gap = 1.0 - (
        abs(eigvals[r]) / abs(eigvals[r - 1])
    )

    M = r + 1 if gap <= t else r

    # 4. Eigenvalue-weighted eigenvector ratios
    eta = U[:, :M] * eigvals[:M]

    denominator = eta[:, 0]

    

    R = eta[:, 1:] / denominator[:, None]

    

    # 5. KMeans
    labels = KMeans(
        n_clusters=r,
        n_init=n_init,
        random_state=init_seed
    ).fit_predict(R)

    return labels