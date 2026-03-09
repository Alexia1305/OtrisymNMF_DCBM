import numpy as np
import networkx as nx
import pysbm
from dcbm import dcbm
import otrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time


def sample_from_P(P):
    n = P.shape[1]
    A = np.zeros((n, n))
    i, j = np.triu_indices(n, 1)
    A[i, j] = (np.random.rand(len(i)) < P[i, j]).astype(int)
    A = A + A.T
    return A


def generate_graph(n=300, r=3, beta=2.5, ave_deg=20, p=1, q=0.2):
    # power_law
    xmin = 1

    # Creation of Z matrice
    labels = np.repeat(np.arange(r), int(n / r))
    np.random.shuffle(labels)
    Z = np.zeros((n, r))
    Z[np.arange(n), labels] = 1
    # Degree parameters from a power law distribution
    u = np.random.rand(n)
    w = xmin * (1 - u) ** (-1 / (beta - 1))
    #w = np.random.exponential(size=n) + 0.2
    w_normalized = (w / (Z @ (Z.T @ w) / (n / r))).flatten()
    Z[np.arange(n), labels] = w_normalized

    # Creation of theta
    theta = (p - q) * np.eye(r) + q

    P = (Z @ theta) @ Z.T
    P = (ave_deg * n / np.sum(P)) * P

    A = sample_from_P(P)
    # no isolated nodes
    indices = np.where(A.sum(axis=1) != 0)[0]
    A = A[np.ix_(indices, indices)]
    labels = labels[indices]
    G = nx.from_numpy_array(A)
    print(2 * G.number_of_edges() / G.number_of_nodes())
    return G, labels


def test():
    n = 300
    r = 3
    beta = 3.5
    ave_deg = 30
    p = 0.1
    q = 0.02
    nbr_tests = 10
    trials = 3
    results = {
        "FROST": {"NMI": [], "Time": []},
        "KL_EM": {"NMI": [], "Time": []},
        "KN": {"NMI": [], "Time": []},
        "MH": {"NMI": [], "Time": []},
        "FROST_SVCA": {"NMI": [], "Time": []},
        "KL_EM_SVCA": {"NMI": [], "Time": []},
        "KN_SVCA": {"NMI": [], "Time": []},
        "MH_SVCA": {"NMI": [], "Time": []},
        "SVCA": {"NMI": [], "Time": []},

    }
    for itt in range(nbr_tests):
        graph, labels = generate_graph(n=n, r=r, beta=beta, ave_deg=ave_deg, p=p, q=q)
        if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
            print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

        # # KL_EM
        # start_time = time.time()
        # EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
        #                     init_method="random", verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(labels, EM_partition)
        # results["KL_EM"]["NMI"].append(NMI)
        # results["KL_EM"]["Time"].append(end_time-start_time)


        # KN
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                            numTrials=trials,
                            init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        print("KN", NMI)
        results["KN"]["NMI"].append(NMI)
        results["KN"]["Time"].append(end_time - start_time)


        # # MH
        # start_time = time.time()
        # EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK, numTrials=trials,
        #                     init_method="random", verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(labels, EM_partition)
        # results["MH"]["NMI"].append(NMI)
        # results["MH"]["Time"].append(end_time - start_time)


        # FROST
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="random",
                                                                                  numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, v_best)
        print("FRost", NMI)
        results["FROST"]["NMI"].append(NMI)
        results["FROST"]["Time"].append(end_time - start_time)


        # # KL_EM initialized by SVCA
        # start_time = time.time()
        # EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
        #                     numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(labels, EM_partition)
        # results["KL_EM_SVCA"]["NMI"].append(NMI)
        # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)


        # KN initialized by SVCA
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                            numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        print("KN_SVCA", NMI)
        results["KN_SVCA"]["NMI"].append(NMI)
        results["KN_SVCA"]["Time"].append(end_time - start_time)

        # # MH initialized by SVCA
        # start_time = time.time()
        # EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK,
        #                     numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(labels, EM_partition)
        # results["MH_SVCA"]["NMI"].append(NMI)
        # results["MH_SVCA"]["Time"].append(end_time - start_time)

        #
        # FROST initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                  numTrials=trials, verbosity=0,
                                                                                  init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, v_best)
        print("Frost_svca", NMI)
        results["FROST_SVCA"]["NMI"].append(NMI)
        results["FROST_SVCA"]["Time"].append(end_time - start_time)


        # SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, v_best)
        print("SVCA", NMI)
        results["SVCA"]["NMI"].append(NMI)
        results["SVCA"]["Time"].append(end_time - start_time)

    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']), 4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1), 4)},Time Mean: {np.round(np.mean(data['Time']), 4)}, Time Std: {np.round(np.std(data['Time'], ddof=1), 4)}")

    with open('polblogs_results.txt', 'w') as file:
        for algo, data in results.items():
            # Calcul des statistiques
            nmi_mean = np.mean(data['NMI'])
            nmi_std = np.std(data['NMI'], ddof=1)

            # Enregistrer les résultats dans le fichier texte
            file.write(
                f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']), 4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1), 4)},Time Mean: {np.round(np.mean(data['Time']), 4)}, Time Std: {np.round(np.std(data['Time'], ddof=1), 4)}\n")


print('ok')


if __name__ == "__main__":
    test()
