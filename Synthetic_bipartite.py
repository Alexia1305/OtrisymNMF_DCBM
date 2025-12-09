import numpy as np
import scipy.sparse as sp
from scipy.stats import poisson
from matplotlib import pyplot as plt
from scipy.sparse import diags

import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import pysbm
import time

import pandas as pd
from Utils import DC_BM
import random

def Generate_difficult_networks(L,nbr_graph):
    # Networks Parameters
    # Number of nodes
    N = 1000
    # Node types
    types = np.ones(N, dtype=int)
    types[700:] = 2

    # Correct planted communities (g)
    g = np.zeros(N, dtype=int)

    g[:350] = 1
    g[350:700] = 2
    g[700:800] = 3
    g[800:950] = 4
    g[950:1000] = 5

    # Member list of each community
    members = [np.where(g == j)[0] for j in range(1, 6)]

    # ------------------------------------------------------------
    # Block structure matrix s
    # ------------------------------------------------------------
    s = np.zeros((5, 5))
    s[0, 2] = 2500
    s[1, 3] = 2500
    s[0, 4] = 1500
    s[1, 4] = 1500
    s = s + s.T

    # ------------------------------------------------------------
    # Edge affinities (d vector)
    # ------------------------------------------------------------
    d = 15 * (2 * np.random.rand(N) // 1) + 10  # like 15*floor(2*rand)+10

    for i in range(1, 6):
        idx = np.where(g == i)[0]
        d[idx] = d[idx] / np.sum(d[idx])

    # ------------------------------------------------------------
    # Number of edges per community
    # ------------------------------------------------------------
    k = np.sum(s, axis=1)
    m = np.sum(k) / 2

    # ------------------------------------------------------------
    # Null model (n matrix)
    # ------------------------------------------------------------
    n = np.zeros((5, 5))
    for i in [0, 1]:
        for j in [2, 3, 4]:
            n[i, j] = k[i] * k[j] / m

    n = n + n.T



    # Storage
    A = []
    labels = []
    # ------------------------------------------------------------
    # Generate networks
    # ------------------------------------------------------------
    for elem in range(nbr_graph):
        # mixture of planted/random structure
        mix = L * s + (1 - L) * n
        while True:
            # draw Poisson edge counts
            w = np.zeros((5, 5), dtype=int)
            for i in [0, 1]:
                for j in [2, 3, 4]:
                    w[i, j] = poisson.rvs(mix[i, j])

            # create edges
            rows = []
            cols = []

            for i in [0, 1]:
                for j in [2, 3, 4]:

                    # Cumulative affinities for choosing vertices
                    d_i = d[members[i]]
                    d_j = d[members[j]]

                    cum_i = np.cumsum(d_i)
                    cum_j = np.cumsum(d_j)

                    # draw vertex endpoints
                    dice_i = np.random.rand(w[i, j])
                    dice_j = np.random.rand(w[i, j])

                    R = [members[i][np.searchsorted(cum_i, x)] for x in dice_i]
                    C = [members[j][np.searchsorted(cum_j, x)] for x in dice_j]

                    rows.extend(R)
                    cols.extend(C)

            # Sparse adjacency matrix
            A_l = sp.csr_matrix(
                (np.ones(len(rows)), (rows, cols)),
                shape=(N, N)
            )

            # Symmetrize A
            A_l = A_l + A_l.T

            if not any(A_l.indptr[i] == A_l.indptr[i + 1] for i in range(A_l.shape[0])):
                break


        perm = np.random.permutation(A_l.shape[0])
        A_new = A_l[perm][:, perm]
        g_new = g[perm]
        A.append(A_new)
        labels.append(g_new)
    print(f"Network created. lambda = {L:.2f}")

    return A, labels

def test():
    algorithms = ["OtrisymNMF", "OtrisymNMF_SVCA", "SVCA", "KN", "KN_SVCA"]
    n = 1000
    n_tests = 2
    list_lambda = np.arange(0.7, 1, 0.1)
    results_total = {
        "OtrisymNMF": {"NMI": [], "AMI": [], "Time": []},
        "KN": {"NMI": [], "AMI": [], "Time": []},
        "KL_EM": {"NMI": [], "AMI": [], "Time": []},
        "MHA250k": {"NMI": [], "AMI": [], "Time": []},
        "OtrisymNMF_SVCA": {"NMI": [], "AMI": [], "Time": []},
        "SVCA": {"NMI": [], "AMI": [], "Time": []},
        "KN_SVCA": {"NMI": [], "AMI": [], "Time": []},
        "KL_EM_SVCA": {"NMI": [], "AMI": [], "Time": []},
        "MHA250k_SVCA": {"NMI": [], "AMI": [], "Time": []},

    }
    for L in list_lambda:

        A, g = Generate_difficult_networks(L, n_tests)
        results = {
            "OtrisymNMF": {"NMI": [], "AMI": [], "Time": []},
            "KN": {"NMI": [], "AMI": [], "Time": []},
            "KL_EM": {"NMI": [], "AMI": [], "Time": []},
            "MHA250k": {"NMI": [], "AMI": [], "Time": []},
            "OtrisymNMF_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "SVCA": {"NMI": [], "AMI": [], "Time": []},
            "KN_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "KL_EM_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "MHA250k_SVCA": {"NMI": [], "AMI": [], "Time": []},

        }

        for idx, X in enumerate(A, start=1):
            print(f"Processed {idx} out of {len(A)} graphs.")

            labels = g[idx-1]
            r = max(labels)
            G = nx.from_scipy_sparse_matrix(X.astype(int))


            # OtrisymNMF

            start_time = time.time()
            w_best, v_best, S_best, error_best, _ = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=10,
                                                                             init_method="random",
                                                                             verbosity=0, init_seed=idx,
                                                                             delta=1e-5, )
            end_time = time.time()
            results["OtrisymNMF"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["OtrisymNMF"]["AMI"].append(adjusted_mutual_info_score(labels, v_best, average_method='max'))
            results["OtrisymNMF"]["Time"].append(end_time - start_time)

            # KN
            start_time = time.time()
            KN_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=10,
                                    init_method="random", verbosity=0)
            end_time = time.time()
            results["KN"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            results["KN"]["Time"].append(end_time - start_time)


            # # KL_EM
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="random", verbosity=0)
            # end_time = time.time()
            # results["KL_EM"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            # results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            # results["KL_EM"]["Time"].append(end_time - start_time)
            #
            #
            # # MHA250
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,init_method="random",
            #                       verbosity=0)
            # end_time = time.time()
            #
            # results["MHA250k"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            # results["MHA250k"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            # results["MHA250k"]["Time"].append(end_time - start_time)

            # OtrisymNMF_SVCA

            start_time = time.time()
            w_best, v_best, S_best, error_best, _ = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=10, init_method="SVCA",
                                                                             verbosity=0, init_seed=idx, delta=1e-5)
            end_time = time.time()

            results["OtrisymNMF_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["OtrisymNMF_SVCA"]["AMI"].append(
                adjusted_mutual_info_score(labels, v_best, average_method='max'))
            results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)

            # SVCA only
            start_time = time.time()
            w_best, v, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=10, verbosity=0)
            end_time = time.time()
            results["SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v))
            results["SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v, average_method='max'))
            results["SVCA"]["Time"].append(end_time - start_time)

            # # KL_EM initialized by SVCA
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # results["KL_EM_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            # results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            #
            # KN initialized by SVCA
            start_time = time.time()
            KN_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=10,
                                 init_method="SVCA", verbosity=0, init_seed=idx)
            end_time = time.time()
            results["KN_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            results["KN_SVCA"]["Time"].append(end_time - start_time)


            # # MHA250 initialized by SVCA
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,
            #                       init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # results["MHA250k_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            # results["MHA250k_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            # results["MHA250k_SVCA"]["Time"].append(end_time - start_time)

        for algo in algorithms:
            mean_NMI = np.mean(results[algo]["NMI"])
            mean_Time = np.mean(results[algo]["Time"])

            results_total[algo]["NMI"].append(mean_NMI)
            results_total[algo]["Time"].append(mean_Time)

            print(f"{algo}: NMI_mean={mean_NMI:.4f},  Time_mean={mean_Time:.2f}s")

if __name__ == "__main__":
    test()
