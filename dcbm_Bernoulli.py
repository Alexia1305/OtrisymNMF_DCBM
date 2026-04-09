import numpy as np
import networkx as nx
import pysbm
from dcbm import dcbm
import otrisymNMF
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import random, time
import clustering_mi as cmi  # for the assymetrically normalized of the reduced mutual information


def sample_from_P(P):
    n = P.shape[1]
    A = np.zeros((n, n))
    i, j = np.triu_indices(n, 1)
    A[i, j] = (np.random.rand(len(i)) < P[i, j]).astype(int)
    A = A + A.T
    return A


def sample_powerlaw(n, beta, kmin=2, kmax=30):
    u = np.random.rand(n)
    return (kmin ** (1 - beta) + u * (kmax ** (1 - beta) - kmin ** (1 - beta))) ** (1 / (1 - beta))


def generate_graph(n=300, r=3, beta=2.5, ave_deg=20, p=1, q=0.2):
    # power_law
    xmin = 1

    # Creation of Z matrice
    labels = np.repeat(np.arange(r), int(n / r))
    np.random.shuffle(labels)
    Z = np.zeros((n, r))
    Z[np.arange(n), labels] = 1
    # Degree parameters from a power law distribution
    w = sample_powerlaw(n, beta)
    # w = np.random.exponential(size=n) + 0.2
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
    ##print(2 * G.number_of_edges() / G.number_of_nodes())
    return G, labels


def generate_graph_noise(n=300, r=3, beta=2.5, ave_deg=20, p=1, q=0.2, epsi=0):
    # power_law
    xmin = 1

    # Creation of Z matrice
    labels = np.repeat(np.arange(r), int(n / r))
    np.random.shuffle(labels)
    Z = np.zeros((n, r))
    Z[np.arange(n), labels] = 1
    # Degree parameters from a power law distribution
    w = sample_powerlaw(n, beta)
    # w = np.random.exponential(size=n) + 0.2
    w_normalized = (w / (Z @ (Z.T @ w) / (n / r))).flatten()
    Z[np.arange(n), labels] = w_normalized

    # Creation of theta
    theta = (p - q) * np.eye(r) + q

    P = (Z @ theta) @ Z.T
    P = (ave_deg * n / np.sum(P)) * P

    A = sample_from_P(P)
    noise = np.random.binomial(1, epsi, size=A.shape)
    noise = np.triu(noise, 1)
    noise = noise + noise.T
    A = np.logical_xor(A, noise).astype(int)

    # A = np.minimum(A + noise, 1)
    # no isolated nodes
    indices = np.where(A.sum(axis=1) != 0)[0]
    A = A[np.ix_(indices, indices)]
    labels = labels[indices]
    G = nx.from_numpy_array(A)
    ##print(2 * G.number_of_edges() / G.number_of_nodes())
    return G, labels


def testbeta(n=600, r=3, p=0.9, q=0.3, ave_deg=20, list_beta=[2.5, 2.75, 3, 3.25, 3.5]):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 50
    trials = 1
    with open("test_beta_results.txt", "w") as file:
        for beta in list_beta:
            print(f"\nRunning experiments for mu = {beta}\n")
            file.write(f"\n===== beta = {beta} =====\n")

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

                # KL_EM
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM", NMI)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["Time"].append(end_time - start_time)

                # KN
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KN", NMI)
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
                # #print("FRost", NMI)
                results["FROST"]["NMI"].append(NMI)
                results["FROST"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                ##print("KN_SVCA", NMI)
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
                ##print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["Time"].append(end_time - start_time)

                # SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials,
                                                                                         verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # #print("SVCA", NMI)
                results["SVCA"]["NMI"].append(NMI)
                results["SVCA"]["Time"].append(end_time - start_time)

            for algo, data in results.items():
                nmi_mean = np.mean(data["NMI"])
                nmi_std = np.std(data["NMI"], ddof=1)

                time_mean = np.mean(data["Time"])
                time_std = np.std(data["Time"], ddof=1)

                line = (
                    f"Algorithm: {algo}, "
                    f"NMI Mean: {np.round(nmi_mean, 4)}, "
                    f"NMI Std: {np.round(nmi_std, 4)}, "
                    f"Time Mean: {np.round(time_mean, 4)}, "
                    f"Time Std: {np.round(time_std, 4)}"
                )

                # print(line)
                file.write(line + "\n")


def testave_deg(n=600, r=3, p=0.9, q=0.3, list_ave_deg=[10, 20, 40, 60], beta=3):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 50
    trials = 1
    with open("test_avedeg_results.txt", "w") as file:
        for ave_deg in list_ave_deg:
            print(f"\nRunning experiments for mu = {beta}\n")
            file.write(f"\n===== alpha = {ave_deg} =====\n")

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

                # KL_EM
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM", NMI)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["Time"].append(end_time - start_time)

                # KN
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KN", NMI)
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
                # #print("FRost", NMI)
                results["FROST"]["NMI"].append(NMI)
                results["FROST"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                ##print("KN_SVCA", NMI)
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
                ##print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["Time"].append(end_time - start_time)

                # SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials,
                                                                                         verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # #print("SVCA", NMI)
                results["SVCA"]["NMI"].append(NMI)
                results["SVCA"]["Time"].append(end_time - start_time)

            for algo, data in results.items():
                nmi_mean = np.mean(data["NMI"])
                nmi_std = np.std(data["NMI"], ddof=1)

                time_mean = np.mean(data["Time"])
                time_std = np.std(data["Time"], ddof=1)

                line = (
                    f"Algorithm: {algo}, "
                    f"NMI Mean: {np.round(nmi_mean, 4)}, "
                    f"NMI Std: {np.round(nmi_std, 4)}, "
                    f"Time Mean: {np.round(time_mean, 4)}, "
                    f"Time Std: {np.round(time_std, 4)}"
                )

                # print(line)
                file.write(line + "\n")


def testSNR(n=600, r=3, p=1, q_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ave_deg=20, beta=3):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 50
    trials = 1
    with open("test_SNR_results.txt", "w") as file:
        for q in q_list:
            print(f"\nRunning experiments for q = {q}\n")
            file.write(f"\n===== = {q} =====\n")

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

                # KL_EM
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM", NMI)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["Time"].append(end_time - start_time)

                # KN
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KN", NMI)
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
                # #print("FRost", NMI)
                results["FROST"]["NMI"].append(NMI)
                results["FROST"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                ##print("KN_SVCA", NMI)
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
                ##print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["Time"].append(end_time - start_time)

                # SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials,
                                                                                         verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # #print("SVCA", NMI)
                results["SVCA"]["NMI"].append(NMI)
                results["SVCA"]["Time"].append(end_time - start_time)

            for algo, data in results.items():
                nmi_mean = np.mean(data["NMI"])
                nmi_std = np.std(data["NMI"], ddof=1)

                time_mean = np.mean(data["Time"])
                time_std = np.std(data["Time"], ddof=1)

                line = (
                    f"Algorithm: {algo}, "
                    f"NMI Mean: {np.round(nmi_mean, 4)}, "
                    f"NMI Std: {np.round(nmi_std, 4)}, "
                    f"Time Mean: {np.round(time_mean, 4)}, "
                    f"Time Std: {np.round(time_std, 4)}"
                )

                # print(line)
                file.write(line + "\n")


def testNoise(n=600, r=3, p=0.9, q=0.3, ave_deg=20, beta=2.5):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 100
    trials = 10
    epsilons = np.arange(0.05, 0.06, 0.01)
    with open("results/test_Noise_results.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r}, p:{p}, q:{q}, ave_deg:{ave_deg}, beta:{beta}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for epsi in epsilons:
            print(f"\nRunning experiments for epsi = {epsi}\n")
            file.write(f"\n===== = {epsi} =====\n")

            results = {
                "FROST": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "MH": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "MH_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN(T)": {"NMI": [], "AMI": [], "RMI": [], "Time": []},

            }
            for itt in range(nbr_tests):
                graph, labels = generate_graph_noise(n=n, r=r, beta=beta, ave_deg=ave_deg, p=p, q=q, epsi=epsi)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

                # KL_EM
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # print("KL_EM", NMI)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM"]["Time"].append(end_time - start_time)

                # KN
                start_time = time.time()
                KN_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN", NMI)
                results["KN"]["NMI"].append(NMI)
                results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN"]["Time"].append(end_time - start_time)
                # print("KN time",end_time - start_time)

                # MH
                start_time = time.time()
                MH_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
                                    pysbm.MetropolisHastingInferenceTenK, numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, MH_partition)
                results["MH"]["NMI"].append(NMI)
                results["MH"]["AMI"].append(adjusted_mutual_info_score(labels, MH_partition))
                results["MH"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, MH_partition, variation="reduced", normalization="first"))
                results["MH"]["Time"].append(end_time - start_time)
                # print("MH",NMI)
                # print("MH time", end_time - start_time)

                # FROST
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="random",
                                                                                          numTrials=trials, verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # print("FRost", NMI)
                results["FROST"]["NMI"].append(NMI)
                results["FROST"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by SVCA
                start_time = time.time()
                KN_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by true labels
                start_time = time.time()
                KN_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_partition=labels, verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN(T)", NMI)
                results["KN(T)"]["NMI"].append(NMI)
                results["KN(T)"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN(T)"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN(T)"]["Time"].append(end_time - start_time)

                # MH initialized by SVCA
                start_time = time.time()
                MH_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
                                    pysbm.MetropolisHastingInferenceTenK,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, MH_partition)
                results["MH_SVCA"]["NMI"].append(NMI)
                results["MH_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, MH_partition))
                results["MH_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, MH_partition, variation="reduced", normalization="first"))
                results["MH_SVCA"]["Time"].append(end_time - start_time)
                # print("MH",NMI)

                #
                # FROST initialized by SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                          numTrials=trials, verbosity=0,
                                                                                          init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST_SVCA"]["Time"].append(end_time - start_time)

                # SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials,
                                                                                         verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                # print("SVCA", NMI)
                results["SVCA"]["NMI"].append(NMI)
                results["SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["SVCA"]["Time"].append(end_time - start_time)

            for algo, data in results.items():
                nmi_mean = np.mean(data["NMI"])
                nmi_std = np.std(data["NMI"], ddof=1)
                ami_mean = np.mean(data["AMI"])
                ami_std = np.std(data["AMI"], ddof=1)
                rmi_mean = np.mean(data["RMI"])
                rmi_std = np.std(data["RMI"], ddof=1)

                time_mean = np.mean(data["Time"])
                time_std = np.std(data["Time"], ddof=1)

                line = (
                    f"Algorithm: {algo}, "
                    f"NMI Mean: {np.round(nmi_mean, 4)}, "
                    f"NMI Std: {np.round(nmi_std, 4)}, "
                    f"AMI Mean: {np.round(ami_mean, 4)}, "
                    f"AMI Std: {np.round(ami_std, 4)}, "
                    f"RMI Mean: {np.round(rmi_mean, 4)}, "
                    f"RMI Std: {np.round(rmi_std, 4)}, "
                    f"Time Mean: {np.round(time_mean, 4)}, "
                    f"Time Std: {np.round(time_std, 4)}"
                )

                print(line)
                file.write(line + "\n")


if __name__ == "__main__":
    testNoise()
