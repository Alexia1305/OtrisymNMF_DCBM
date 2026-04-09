import numpy as np
import networkx as nx
import pysbm
from dcbm import dcbm
import otrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time


def sample_from_P(P):
    n = P.shape[0]
    U = np.random.rand(n, n)
    A = (U < P).astype(int)
    A = np.triu(A, 1)
    A = A + A.T
    return A

def sample_powerlaw(n, beta, kmin=1, kmax=30):
    u = np.random.rand(n)
    return (kmin**(1-beta) + u*(kmax**(1-beta)-kmin**(1-beta)))**(1/(1-beta))


def generate_DCBM(g,theta,w):
    r = int(np.max(g)) + 1
    n = len(g)
    A = np.zeros((n, n))
    w_s = np.zeros((r, r))
    for c1 in range(r):
        for c2 in range(c1+1):
            if c1 == c2:
                w_s[c1, c2] = np.random.poisson(lam=w[c1, c2] // 2)
            else:
                w_s[c1, c2] = np.random.poisson(lam=w[c1, c2])
                w_s[c2, c1] = w_s[c1, c2]
            for edge in range(int(w_s[c1, c2])):
                nodes_c1 = np.where(g == c1)[0]
                nodes_c2 = np.where(g == c2)[0]
                i = np.random.choice(nodes_c1, p=theta[nodes_c1])
                j = np.random.choice(nodes_c2, p=theta[nodes_c2])
                A[i, j] += 1
                A[j, i] += 1
    return A








def network_AS(lamb, seed):
    n = 1000
    np.random.seed(seed)
    k = np.random.choice([10, 30], size=n) #node degree
    m = np.sum(k)/2
    g = np.random.choice([0, 1], size=n)
    K = np.zeros(2)
    K[0] = np.sum(k[g == 0])
    K[1] = np.sum(k[g == 1])
    w_ptd = np.diag(K)
    w_rdm = np.array([[(K[0]**2)/(2*m), (K[0]*K[1])/(2*m)], [(K[0]*K[1])/(2*m), (K[1]**2)/(2*m)]])

    w = lamb*w_ptd+(1-lamb)*w_rdm
    theta = np.zeros(n)
    theta[g == 0] = k[g == 0] / K[0]
    theta[g == 1] = k[g == 1] / K[1]
    A = generate_DCBM(g, theta, w)
    # no isolated nodes
    indices = np.where(A.sum(axis=1) != 0)[0]
    A = A[np.ix_(indices, indices)]
    g = g[indices]
    G = nx.from_numpy_array(A)
    return G,g

def network_CP(lamb, seed):
    n = 1000
    np.random.seed(seed)
    xmin = 10  # minimum degree
    beta = 2.5  # exponent
    u = np.random.rand(n)
    degrees = xmin * (1 - u) ** (-1 / (beta - 1))
    k = degrees.astype(int)
    m = np.sum(k)/2
    g = np.random.choice([0, 1], size=n)
    K = np.zeros(2)
    K[0] = np.sum(k[g == 0])
    K[1] = np.sum(k[g == 1])
    if K[1] > K[0]:
        K1 = K[1]
        K2 = K[0]
    else:
        K1 = K[0]
        K2 = K[1]
    w_ptd = np.array([[abs(K1-K2), K2], [K2, 0]])
    w_rdm = np.array([[(K[0]**2)/(2*m), (K[0]*K[1])/(2*m)], [(K[0]*K[1])/(2*m), (K[1]**2)/(2*m)]])
    w = lamb*w_ptd+(1-lamb)*w_rdm
    theta = np.zeros(n)
    theta[g == 0] = k[g == 0] / K[0]
    theta[g == 1] = k[g == 1] / K[1]
    A = generate_DCBM(g, theta, w)
    # no isolated nodes
    indices = np.where(A.sum(axis=1) != 0)[0]
    A = A[np.ix_(indices, indices)]
    g = g[indices]
    G = nx.from_numpy_array(A)
    return G,g


def network_H(lamb,seed):
    n = 1000
    np.random.seed(seed)
    xmin = 10  # minimum node degree
    beta = 2.5  # exponent
    u = np.random.rand(n)
    degrees = xmin * (1 - u) ** (-1 / (beta - 1))
    k = degrees.astype(int)
    m = np.sum(k)/2
    #communities
    g = np.zeros(n)
    indices = np.random.choice(n, size=500, replace=False)
    g[indices] = 0
    others = np.setdiff1d(np.arange(n), indices)
    g[others] = np.random.choice([1, 2], size=len(others))
    K = np.zeros(3)
    K[0] = np.sum(k[g == 0])
    K[1] = np.sum(k[g == 1])
    K[2] = np.sum(k[g == 2])
    A = (1/4)*min(K[0],K[1])

    w_ptd = np.array([[K[0]-A, A, 0],[A, K[1]-A, 0], [0, 0, K[2]]])
    w_rdm = np.array([[(K[0]**2)/(2*m), (K[0]*K[1])/(2*m), (K[0]*K[2])/(2*m)], [(K[1]*K[0])/(2*m), (K[1]**2)/(2*m), (K[1]*K[2])/(2*m)],[(K[2]*K[0])/(2*m), (K[2]*K[1])/(2*m), (K[2]**2)/(2*m)]])
    w = lamb*w_ptd+(1-lamb)*w_rdm
    theta = np.zeros(n)
    theta[g == 0] = k[g == 0] / K[0]
    theta[g == 1] = k[g == 1] / K[1]
    theta[g == 2] = k[g == 2] / K[2]
    A = generate_DCBM(g, theta, w)
    # no isolated nodes
    indices = np.where(A.sum(axis=1) != 0)[0]
    A = A[np.ix_(indices, indices)]
    g = g[indices]
    G = nx.from_numpy_array(A)
    return G, g.astype(int)



def test(config="AS"):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 30
    trials = 1
    with open(f"test_{config}.txt", "w") as file:
        for lbd in  np.arange(0, 1.1, 0.1):
            print(f"\nRunning experiments for lambda= {lbd}\n")
            file.write(f"\n===== lambda = {lbd} =====\n")

            results = {
                "FROST": {"NMI": [], "Time": []},
                "KL_EM": {"NMI": [], "Time": []},
                "KN": {"NMI": [], "Time": []},
                "FROST_SVCA": {"NMI": [], "Time": []},
                "KL_EM_SVCA": {"NMI": [], "Time": []},
                "KN_SVCA": {"NMI": [], "Time": []},
                "SVCA": {"NMI": [], "Time": []},
                "KN_g": {"NMI": [], "Time": []}

            }
            for itt in range(nbr_tests):
                if config == "AS":
                    graph, labels = network_AS(lbd, seed=itt)
                elif config == "CP":
                    graph, labels = network_CP(lbd, seed=itt)
                elif config == "H":
                    graph, labels = network_H(lbd, seed=itt)

                r = np.max(labels)+1
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

                # KL_EM
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
                                    init_method="random", verbosity=0)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                print("KL_EM", NMI)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["Time"].append(end_time-start_time)


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


                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)


                # KN initialized by SVCA
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KN initialized by the true labels
                start_time = time.time()
                EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_partition=labels, verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                # print("KN_SVCA", NMI)
                results["KN_g"]["NMI"].append(NMI)
                results["KN_g"]["Time"].append(end_time - start_time)

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

                print(line)
                file.write(line + "\n")


if __name__ == "__main__":
    for config in ["H"]:
        test(config)
