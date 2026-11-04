import numpy as np
import networkx as nx
import pysbm
from other_code.spectral_methods import SCORE,SCORE_plus
from other_code.dcbm_inference import dcbm
from other_code.graph_tool import graph_tool_dcbm
import otrisymNMF
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import random, time
import clustering_mi as cmi  # for the assymetrically normalized of the reduced mutual information
from scipy.sparse import csr_matrix,csc_matrix
import scipy.sparse as sp

def sample_from_P(P):
    n = P.shape[1]
    A = np.zeros((n, n))
    i, j = np.triu_indices(n, 1)
    A[i, j] = (np.random.rand(len(i)) < P[i, j]).astype(int)
    A = A + A.T

    return A

def sample_powerlaw(n, gamma, kmin=1):
    u = 1.0- np.random.rand(n)
    return kmin * u ** (-1 / (gamma - 1))

def generate_graph(n=300, r=3, gamma=2.5, ave_deg=10,
                   beta=6,seed=None):
    if seed is not None:
        np.random.seed(seed)
    if not 1 <= r <= n:
        raise ValueError("Il faut 1 <= r <= n.")

    # Balanced communities
    labels = np.arange(n) % r
    np.random.shuffle(labels)
    Z = np.zeros((n, r))
    Z[np.arange(n), labels] = 1

   # Degree parameters
    w = sample_powerlaw(n, gamma)
    community_sizes = Z.sum(axis=0)
    mean_w = (Z.T @ w) / community_sizes
    w_normalized = w / mean_w[labels]

    Z[np.arange(n), labels] = w_normalized

    theta = (beta - 1) * np.eye(r) + 1

    P = (Z @ theta) @ Z.T
    np.fill_diagonal(P, 0)
    P = (ave_deg * n / np.sum(P)) * P

    A = sample_from_P(P)
    degrees = np.asarray(A.sum(axis=1)).ravel()
    keep = degrees > 0
    # Remove isolated nodes 
    A = csr_matrix(A[keep][:, keep])
    labels= np.asarray(labels)[keep]

    G = nx.from_numpy_array(A)

    
    return G, labels

def add_false_edges(A, rho, seed=None):
    """
    Add edges (false positive) to a graph

    rho  :number of added edges / number of initial edges.
    Return the new adjacency matrix 
    """
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho >=0")

    A = csc_matrix(A, dtype=float, copy=True)
    A.eliminate_zeros()
    n = A.shape[0]

    # each edges is represented by a pair (i,j) with i < j.
    upper = sp.triu(A, k=1, format="coo")
    occupied = set(zip(upper.row.tolist(), upper.col.tolist()))
    m0 = len(occupied)
    n_add = int(np.floor(rho * m0))

    if n_add > n * (n - 1) // 2 - m0:
        raise ValueError("not enough non edges")
    if n_add == 0:
        return A

    rng = np.random.default_rng(seed)
    added = []

    while len(added) < n_add:
       
        candidates = rng.integers(0, n, size=(4096, 2))

        for u, v in candidates:
            if u == v:
                continue

            i, j = sorted((int(u), int(v)))

            if (i, j) in occupied:
                continue

            occupied.add((i, j))
            added.append((i, j))

            if len(added) == n_add:
                break

    i, j = np.asarray(added).T
    noise = sp.coo_matrix(
        (
            np.ones(2 * n_add),
            (np.concatenate([i, j]), np.concatenate([j, i])),
        ),
        shape=A.shape,
    ).tocsc()

    return (A + noise).tocsc()

def generate_graph_noisy(n=300, r=3, gamma=2.5, ave_deg=10, beta=6, rho=0, seed= None):
    if seed is not None:
            np.random.seed(seed)
    # Balanced communities
    labels = np.arange(n) % r
    np.random.shuffle(labels)
    Z = np.zeros((n, r))
    Z[np.arange(n), labels] = 1

    # Degree parameters
    w = sample_powerlaw(n, gamma)
    community_sizes = Z.sum(axis=0)
    mean_w = (Z.T @ w) / community_sizes
    w_normalized = w / mean_w[labels]

    Z[np.arange(n), labels] = w_normalized

    theta = (beta - 1) * np.eye(r) + 1

    P = (Z @ theta) @ Z.T
    np.fill_diagonal(P, 0)
    P = (ave_deg * n / np.sum(P)) * P

    A = sample_from_P(P)
    degrees = np.asarray(A.sum(axis=1)).ravel()
    keep = degrees > 0
    # Remove isolated nodes 
    A = csr_matrix(A[keep][:, keep])
    labels= np.asarray(labels)[keep]

    # edges addition 
    A= add_false_edges(A, rho)

    G = nx.from_numpy_array(A)

    return G, labels

"EXP 1: Beta varies, community regimes assortatif and disassortatif"
def EXP1(n=600, r=3, ave_deg=10, gamma=2.5):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 100
    trials = 10
    betas = [0.05, 0.1, 0.25, 0.5, 0.75,1,1.5, 2,2.5, 3, 4, 6, 10, 20]
    
    with open("test_EXP1.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r}, beta: {betas}, ave_deg:{ave_deg}, gamma:{gamma}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for beta in betas:
            print(f"\nRunning experiments for epsi = {beta}\n")
            file.write(f"\n===== = {beta} =====\n")

            results = {
                "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SCORE": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},

            }
            for itt in range(nbr_tests):
                np.random.seed(itt)
                graph, labels = generate_graph(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta,seed=itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

               
                #KN
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN", NMI)
                results["KN"]["NMI"].append(NMI)
                results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN"]["Time"].append(end_time - start_time)
                #print("KN time",end_time - start_time)

                # KL_EM
                start_time = time.time()
                EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM"]["Time"].append(end_time - start_time)

                
                # KN initialized by SVCA
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                               
                # FROST initialized by SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                          numTrials=trials, verbosity=0,
                                                                                          init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                #print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST_SVCA"]["Time"].append(end_time - start_time)
                #print("FROST",NMI)

              
                # SCORE
                start_time = time.time()
                SCORE_partition=SCORE(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE"]["NMI"].append(NMI)
                results["SCORE"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE"]["Time"].append(end_time - start_time)

                # SCORE +
                start_time = time.time()
                SCORE_partition=SCORE_plus(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE+"]["NMI"].append(NMI)
                results["SCORE+"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE+"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE+"]["Time"].append(end_time - start_time)

                #graph-tool
                start_time = time.time()
                graph_tool_partition,obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels,graph_tool_partition)
                results["graph_tool"]["NMI"].append(NMI)
                results["graph_tool"]["AMI"].append(adjusted_mutual_info_score(labels,graph_tool_partition))
                results["graph_tool"]["RMI"].append(
                    cmi.normalized_mutual_information(labels,graph_tool_partition, variation="reduced", normalization="first"))
                results["graph_tool"]["Time"].append(end_time - start_time)
                #print("graph-tool",NMI)


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

"EXP2: Sparsity, average degree varies"
def EXP2(n=600, r=3, beta=6, gamma=2.5):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 100
    trials = 10
    ave_degs= [5,6, 8,10,15,20]
    with open("test_EXP2.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r},  beta: {beta}, ave_deg:{ave_degs}, gamma:{gamma}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for ave_deg in ave_degs:
            print(f"\nRunning experiments for degree = {ave_deg}\n")
            file.write(f"\n===== = {ave_deg} =====\n")

            results = {
                           "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "SCORE": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
           
                       }
            for itt in range(nbr_tests):
                np.random.seed(itt)
                graph, labels = generate_graph(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta, seed=itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

                
                #KN
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN", NMI)
                results["KN"]["NMI"].append(NMI)
                results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN"]["Time"].append(end_time - start_time)
                #print("KN time",end_time - start_time)

                # KL_EM
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM"]["Time"].append(end_time - start_time)

                
                # KN initialized by SVCA
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                                
                # FROST initialized by SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                            numTrials=trials, verbosity=0,
                                                                                            init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                #print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST_SVCA"]["Time"].append(end_time - start_time)
                #print("FROST",NMI)

                
                # SCORE
                start_time = time.time()
                SCORE_partition=SCORE(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE"]["NMI"].append(NMI)
                results["SCORE"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE"]["Time"].append(end_time - start_time)

                # SCORE +
                start_time = time.time()
                SCORE_partition=SCORE_plus(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE+"]["NMI"].append(NMI)
                results["SCORE+"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE+"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE+"]["Time"].append(end_time - start_time)

                #graph-tool
                start_time = time.time()
                graph_tool_partition,obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels,graph_tool_partition)
                results["graph_tool"]["NMI"].append(NMI)
                results["graph_tool"]["AMI"].append(adjusted_mutual_info_score(labels,graph_tool_partition))
                results["graph_tool"]["RMI"].append(
                    cmi.normalized_mutual_information(labels,graph_tool_partition, variation="reduced", normalization="first"))
                results["graph_tool"]["Time"].append(end_time - start_time)
                #print("graph-tool",NMI)

        
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

"EXP3: Degree heterogeneity, gamma varies"
def EXP3(n=600, r=3, beta=6, ave_deg=10):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 100
    trials = 10
    gammas= [2,2.25,2.5,2.75,3,3.25,3.5]
    with open("test_EXP3.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r},  beta: {beta}, ave_deg:{ave_deg}, gamma:{gammas}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for gamma in gammas:
            print(f"\nRunning experiments for epsi = {gamma}\n")
            file.write(f"\n===== = {gamma} =====\n")

            results = {
                           "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "SCORE": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                           "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
           
                       }
            for itt in range(nbr_tests):
                np.random.seed(itt)
                graph, labels = generate_graph(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta,seed= itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

                
                #KN
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN", NMI)
                results["KN"]["NMI"].append(NMI)
                results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN"]["Time"].append(end_time - start_time)
                #print("KN time",end_time - start_time)

                # KL_EM
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM"]["Time"].append(end_time - start_time)

                
                # KN initialized by SVCA
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                                
                # FROST initialized by SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                            numTrials=trials, verbosity=0,
                                                                                            init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                #print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST_SVCA"]["Time"].append(end_time - start_time)
                #print("FROST",NMI)

                
                # SCORE
                start_time = time.time()
                SCORE_partition=SCORE(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE"]["NMI"].append(NMI)
                results["SCORE"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE"]["Time"].append(end_time - start_time)

                # SCORE +
                start_time = time.time()
                SCORE_partition=SCORE_plus(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE+"]["NMI"].append(NMI)
                results["SCORE+"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE+"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE+"]["Time"].append(end_time - start_time)

                #graph-tool
                start_time = time.time()
                graph_tool_partition,obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels,graph_tool_partition)
                results["graph_tool"]["NMI"].append(NMI)
                results["graph_tool"]["AMI"].append(adjusted_mutual_info_score(labels,graph_tool_partition))
                results["graph_tool"]["RMI"].append(
                    cmi.normalized_mutual_information(labels,graph_tool_partition, variation="reduced", normalization="first"))
                results["graph_tool"]["Time"].append(end_time - start_time)
                #print("graph-tool",NMI)
            

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

"EXP4: Edges addition, Robustness to perturbation"
def EXP4(n=600, r=3,beta=6, ave_deg=10, gamma=2.5):
    np.random.seed(13)
    random.seed(42)
    nbr_tests = 100
    trials = 10
    rhos = [0,0.1,0.2, 0.4,0.6,0.8,1]
    with open("test_EXP4.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r}, beta: {beta}, ave_deg:{ave_deg}, gamma:{gamma}, rhos:{rhos} nbr test:{nbr_tests}, trials:{trials} =====\n")
        for rho in rhos:
            print(f"\nRunning experiments for epsi = {rho}\n")
            file.write(f"\n===== = {rho} =====\n")

            results = {
                            "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "SCORE": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                            "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
            
                        }
            for itt in range(nbr_tests):
                graph, labels = generate_graph_noisy(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta,rho=rho,seed=itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

               
                #KN
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN", NMI)
                results["KN"]["NMI"].append(NMI)
                results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN"]["Time"].append(end_time - start_time)
                #print("KN time",end_time - start_time)

                # KL_EM
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                results["KL_EM"]["NMI"].append(NMI)
                results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM"]["Time"].append(end_time - start_time)

                
                # KN initialized by SVCA
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, KN_partition)
                # print("KN_SVCA", NMI)
                results["KN_SVCA"]["NMI"].append(NMI)
                results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition))
                results["KN_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
                results["KN_SVCA"]["Time"].append(end_time - start_time)

                # KL_EM initialized by SVCA
                start_time = time.time()
                EM_partition,obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                                    numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, EM_partition)
                #print("KL_EM_SVCA", NMI)
                results["KL_EM_SVCA"]["NMI"].append(NMI)
                results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition))
                results["KL_EM_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

                                
                # FROST initialized by SVCA
                start_time = time.time()
                X = nx.adjacency_matrix(graph)
                w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA",
                                                                                            numTrials=trials, verbosity=0,
                                                                                            init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, v_best)
                #print("Frost_svca", NMI)
                results["FROST_SVCA"]["NMI"].append(NMI)
                results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best))
                results["FROST_SVCA"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
                results["FROST_SVCA"]["Time"].append(end_time - start_time)
                #print("FROST",NMI)

                
                # SCORE
                start_time = time.time()
                SCORE_partition=SCORE(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE"]["NMI"].append(NMI)
                results["SCORE"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE"]["Time"].append(end_time - start_time)

                # SCORE +
                start_time = time.time()
                SCORE_partition=SCORE_plus(X,r,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels, SCORE_partition)
                #print("SCORE",NMI)
                results["SCORE+"]["NMI"].append(NMI)
                results["SCORE+"]["AMI"].append(adjusted_mutual_info_score(labels, SCORE_partition))
                results["SCORE+"]["RMI"].append(
                    cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
                results["SCORE+"]["Time"].append(end_time - start_time)

                #graph-tool
                start_time = time.time()
                graph_tool_partition,obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
                end_time = time.time()
                NMI = normalized_mutual_info_score(labels,graph_tool_partition)
                results["graph_tool"]["NMI"].append(NMI)
                results["graph_tool"]["AMI"].append(adjusted_mutual_info_score(labels,graph_tool_partition))
                results["graph_tool"]["RMI"].append(
                    cmi.normalized_mutual_information(labels,graph_tool_partition, variation="reduced", normalization="first"))
                results["graph_tool"]["Time"].append(end_time - start_time)
                #print("graph-tool",NMI)


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
    
    #EXP 1: Beta varies, community regimes assortatif and disassortatif"
    EXP1()
    #EXP2: Sparsity, average degree varies"
    EXP2()
    #EXP3: Degree heterogeneity, gamma varies"
    EXP3()
    #EXP4: Edges addition, Robustness to perturbation"
    EXP4()

   
   
