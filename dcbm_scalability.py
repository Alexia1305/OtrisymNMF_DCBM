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
import scipy.sparse as sp
import math
import numpy as np
from scipy.sparse import csr_matrix,csr_array

"""Sparse sampling of simple, undirected graphs from a Bernoulli DCBM.
Sampling uses geometric skipping within degree-sorted blocks, adapted from
Miller and Hagberg (2011), Efficient Generation of Networks with Given
Expected Degrees.
"""
def _sample_edges(w, labels, sizes, p, q, rng, expected_edges):
    """Sample independent Bernoulli edges; w, labels and sizes are fixed."""
    n, r = len(w), len(sizes)
    # Sort nodes by community, then by decreasing weight within each community.
    order = np.lexsort((-w, labels))
    ws = w[order]
    starts = np.concatenate(([0], np.cumsum(sizes)))

    # Check the largest off-diagonal probability without constructing P.
    non_singletons = starts[:-1][sizes > 1]
    max_inside = float(np.max(
        p * ws[non_singletons] * ws[non_singletons + 1], initial=0.0
    ))
    max_between = 0.0
    if r > 1:
        largest = np.partition(ws[starts[:-1]], r - 2)[-2:]
        max_between = float(q * largest[1] * largest[0])
    max_probability = max(max_inside, max_between)
    if not np.isfinite(max_probability):
        raise ValueError("Nonfinite edge probabilities; check the weights and parameters.")
    

    # Allocate edge arrays, growing them only if necessary.
    capacity = max(16, int(1.1 * expected_edges) + 64)
    rows = np.empty(capacity, dtype=np.int64)
    cols = np.empty(capacity, dtype=np.int64)
    m = 0
    random = rng.random
    log1p = math.log1p

    for k in range(r):
        for l in range(k, r):
            theta_kl = p if k == l else q
            if theta_kl == 0:
                continue
            stop = int(starts[l + 1])
            for i in range(int(starts[k]), int(starts[k + 1])):
                # Within a community, visit only one triangular half.
                j = i + 1 if k == l else int(starts[l])
                factor = theta_kl * ws[i]
                while j < stop:
                    # Weights are sorted: upper bounds every remaining P[i,j].
                    upper = min(1.0, factor * ws[j])
                    if upper <= 0:
                        break
                    if upper < 1:
                        # Geometric number of absent candidate edges.
                        jump = log1p(-random()) / log1p(-upper)
                        if jump >= stop - j:
                            break
                        j += int(jump)
                    probability = min(1.0, factor * ws[j])
                    # Thin the candidate to its exact Bernoulli probability.
                    if probability >= upper or random() < probability / upper:
                        if m == capacity:
                            capacity *= 2
                            rows.resize(capacity, refcheck=False)
                            cols.resize(capacity, refcheck=False)
                        rows[m], cols[m] = order[i], order[j]
                        m += 1
                    j += 1

    # Every unordered pair was considered once: no duplicates or self-loops.
    row = np.concatenate((rows[:m], cols[:m]))
    col = np.concatenate((cols[:m], rows[:m]))
    A = csr_matrix((np.ones(2 * m), (row, col)), shape=(n, n))
    return A, max_probability


def generate_dcbm_sparse(
    n=3000, r=3, gamma=2.5, ave_deg=10, beta=6,
    kmin=1.0, kmax=None, seed=None
):
    """Generate a sparse Bernoulli DCBM with balanced communities.

    For i < j, independently conditional on weights and labels:
        A[i,j] = A[j,i] ~ Bernoulli(min(1, w[i]*w[j]*theta[labels[i], labels[j]]))
    where theta has diagonal p and off-diagonal q. A[i,i] = 0.

    Parameters
    ----------
    n, r : int
        Number of nodes and communities.
    gamma : float > 1
        Density exponent of the power law for the raw weights: x**(-gamma).
    ave_deg : float
        Target expected average degree BEFORE probability clipping, excluding loops.
    beta : float >= 0
        Ratio p/q
    kmin : positive float
        Lower bound of the raw Pareto weights.
    seed : int

    Returns
    -------
    A : scipy.sparse.csr_array, shape (n, n), float64
        Binary, symmetric adjacency matrix with a zero diagonal.
        Isolated nodes are retained; 
    labels : numpy.ndarray, shape (n,)
        Planted community labels, from 0 to r-1, in the same order as A.

    """
    rng = np.random.default_rng(seed)

    # Balanced communities
    sizes = np.full(r, n // r, dtype=np.int64)
    sizes[:n % r] += 1
    labels = np.repeat(np.arange(r), sizes)
    rng.shuffle(labels)

    # Inverse CDF of the power law
    u = 1.0 - rng.random(n)
    w = kmin * u ** (-1.0 / (gamma - 1.0))
    # normalization by community 
    means = np.bincount(labels, weights=w, minlength=r) / sizes
    w /= means[labels]

    # Sum of w_i*w_j over ordered distinct pairs, inside/outside communities.
    s = np.bincount(labels, weights=w, minlength=r)
    h = np.bincount(labels, weights=w * w, minlength=r)
    U = float(np.sum(s * s - h))
    V = float(s.sum() ** 2 - np.sum(s * s))

    # Calibrate the sum of raw off-diagonal rates is n*ave_deg.
    if ave_deg == 0:
        p = q = 0.0
    
    denominator = beta * U + V
    q = n * ave_deg / denominator
    p = beta * q

    A, max_probability = _sample_edges(
        w, labels, sizes, p, q, rng, n * ave_deg / 2
    )

    # Remove isolated nodes
    keep = np.asarray(A.sum(axis=1)).ravel() > 0
    print(sum(keep))
    A = A[keep][:, keep].tocsr()
    A = csr_array(A)
    labels = labels[keep]
    return A, labels


def EXP_n(gamma=2.5, ave_deg=10, beta=6,r=3):
    nbr_tests = 10
    trials = 1
    n_list =[3000,6000,10000,20000,30000,60000,100000]

    with open("test_EXP_n.txt", "a") as file:
        file.write(f"\n Parameters n: {n_list}, r: {r},  beta: {beta}, ave_deg:{ave_deg}, gamma:{gamma}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for conf, n in enumerate(n_list):
           
            print(f"\nRunning experiments for n = {n}\n")
            file.write(f"\n===== = {n} =====\n")

            results = {
                "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SCORE": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},

            }
            for itt in range(nbr_tests):
                np.random.seed(itt)
                X, labels =generate_dcbm_sparse(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta,
                    kmin=1.0,seed=itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

               
                # #KN
                graph = nx.from_scipy_sparse_array(X)
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
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
                    cmi.normalized_mutual_information(labels, EM_partition,variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

            
            

                
                

                # FROST initialized by SVCA
                start_time = time.time()
                
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
                graph_tool_partition, obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
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

                # Valeurs de chaque test (une valeur par essai, dans l'ordre des tests)
                details = (
                    f"Algorithm: {algo} (per test), "
                    f"NMI: {np.round(data['NMI'], 4).tolist()}, "
                    f"AMI: {np.round(data['AMI'], 4).tolist()}, "
                    f"RMI: {np.round(data['RMI'], 4).tolist()}, "
                    f"Time: {np.round(data['Time'], 4).tolist()}"
                )

                print(line)
                print(details)
                file.write(line + "\n")
                file.write(details + "\n")

def EXP_r(gamma=2.5, ave_deg=10, beta=6,n=30000):
    nbr_tests = 10
    trials = 1
    r_list =[3,5,10,20,30,50]
    betas= 3*(np.array(r_list)-1)

    with open("test_EXP_r.txt", "a") as file:
        file.write(f"\n Parameters n: {n}, r: {r_list},  beta: {betas}, ave_deg:{ave_deg}, gamma:{gamma}, nbr test:{nbr_tests}, trials:{trials} =====\n")
        for conf, r in enumerate(r_list):
            beta=betas[conf]
            print(f"\nRunning experiments for r = {r}\n")
            file.write(f"\n===== = {r} =====\n")

            results = {
                "KN": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM":{"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "FROST_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KN_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "KL_EM_SVCA": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "SCORE+": {"NMI": [], "AMI": [], "RMI": [], "Time": []},
                "graph_tool": {"NMI": [], "AMI": [], "RMI": [], "Time": []},

            }
            for itt in range(nbr_tests):
                np.random.seed(itt)
                X, labels =generate_dcbm_sparse(n=n, r=r, gamma=gamma, ave_deg=ave_deg, beta=beta,
                    kmin=1.0,seed=itt)
                if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
                    print(f"Test completed: {itt / nbr_tests * 100:.0f}%")

                graph = nx.from_scipy_sparse_array(X)
                #KN
                start_time = time.time()
                KN_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=trials,
                                    init_method="random", verbosity=0, init_seed=itt)
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
                    cmi.normalized_mutual_information(labels, EM_partition,variation="reduced", normalization="first"))
                results["KL_EM_SVCA"]["Time"].append(end_time - start_time)

    
                # FROST initialized by SVCA
                start_time = time.time()
                
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
                graph_tool_partition, obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=itt)
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

                # Valeurs de chaque test (une valeur par essai, dans l'ordre des tests)
                details = (
                    f"Algorithm: {algo} (per test), "
                    f"NMI: {np.round(data['NMI'], 4).tolist()}, "
                    f"AMI: {np.round(data['AMI'], 4).tolist()}, "
                    f"RMI: {np.round(data['RMI'], 4).tolist()}, "
                    f"Time: {np.round(data['Time'], 4).tolist()}"
                )

                print(line)
                print(details)
                file.write(line + "\n")
                file.write(details + "\n")

if __name__ == "__main__":
    
    #EXP_r()
    EXP_n()
   
   
