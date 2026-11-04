import graph_tool.all as gt
import numpy as np

import graph_tool.all as gt
import numpy as np


def graph_tool_dcbm(A, r, nTrials=1, init_seed=None):
    """
    Fit a degree-corrected SBM with r communities using multiple runs.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (n, n)
        Symmetric, binary adjacency matrix without self-loops.
    r : int
        Number of communities.
    nTrials : int, default=10
        Number of optimization runs.
    init_seed : int, optional
        Seed used to generate the seed for each run.

    Returns
    -------
    np.ndarray, shape (n,)
        Community labels for the solution with the smallest description
        length, relabeled consecutively starting at zero.
    """
    if not isinstance(nTrials, (int, np.integer)) or nTrials < 1:
        raise ValueError("nTrials must be a positive integer.")

    A = A.tocoo(copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()

    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")
    if not isinstance(r, (int, np.integer)) or not 1 <= r <= n:
        raise ValueError("r must be an integer between 1 and n.")

    # Build the graph once, preserving isolated nodes.
    mask = A.row < A.col
    edges = np.column_stack((A.row[mask], A.col[mask]))

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(edges)

    rng = np.random.default_rng(init_seed)
    best_objective = np.inf
    best_labels = None

    for _ in range(nTrials):
        # Use a different seed for each run.
        trial_seed = int(rng.integers(1, 2**31 - 1))
        gt.seed_rng(trial_seed)

        state = gt.minimize_blockmodel_dl(
            g,
            state_args=dict(deg_corr=True),
            multilevel_mcmc_args=dict(B_min=r, B_max=r),
        )

        objective = float(state.entropy())

        if np.isfinite(objective) and objective < best_objective:
            best_objective = objective
            best_labels = state.get_blocks().get_array().copy()

    if best_labels is None:
        raise RuntimeError("No run produced a finite objective value.")

    # Relabel communities consecutively starting at zero.
    return np.unique(best_labels, return_inverse=True)[1],best_objective