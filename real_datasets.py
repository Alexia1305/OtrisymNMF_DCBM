import numpy as np
import networkx as nx
import otrisymNMF
import pysbm
from sklearn.metrics import normalized_mutual_info_score
import random, time
from other_code.dcbm_inference import dcbm
from other_code.spectral_methods import SCORE_plus
from other_code.graph_tool import graph_tool_dcbm
from scipy.optimize import linear_sum_assignment
import clustering_mi as cmi # for the assymetrically normalized of the reduced mutual information

from scipy.io import loadmat


def load_karate():
    graph = nx.karate_club_graph()
    for u, v in graph.edges():
        graph[u][v].pop('weight', None)  # version unweighted
    club_labels = {node: 1 if graph.nodes[node]['club'] == 'Mr. Hi' else 0 for node in graph.nodes}
    club_vector = np.array([club_labels[node] for node in graph.nodes])

    return graph, club_vector

def load_polblogs():
    G = nx.read_gml('Data/polblogs_cleaned.gml', label='id')
    True_partition = np.asarray([int(G.nodes[node]['value']) for node in G.nodes])

    return G, True_partition

def load_football():
    G = nx.read_gml('Data/football/football.gml', label='id')
    # Supprimer les noeuds dont la valeur est 5
    to_remove = [node for node in G.nodes if int(G.nodes[node]['value']) == 5]
    G.remove_nodes_from(to_remove)

    True_partition = np.asarray([int(G.nodes[node]['value']) for node in G.nodes])
    return G, True_partition

def load_caltech():
    data = loadmat("Data/caltech.mat")
    A = data["A"]
    labels = data["label"]

    print(A.shape)
    print(labels.shape)
    
    G = nx.from_numpy_array(A)
    labels = np.asarray(labels).ravel()
    return G, labels 

def align_and_count_errors(labels, v):
    """Retrun the number of misclassified nodes"""
    labels = np.asarray(labels).ravel()
    v = np.asarray(v).ravel()

    labels_values, labels_idx = np.unique(labels, return_inverse=True)
    v_values, v_idx = np.unique(v, return_inverse=True)

    # Nodes 
    C = np.zeros((len(labels_values), len(v_values)), dtype=np.int64)
    np.add.at(C, (labels_idx, v_idx), 1)

    # Optimal Pertumations 
    row, col = linear_sum_assignment(C, maximize=True)

    if len(col) == len(v_values):
        mapping = np.empty(len(v_values), dtype=labels_values.dtype)
    
    mapping[col] = labels_values[row]
    v_aligned = mapping[v_idx]

    n_errors = int(labels.size - C[row, col].sum())
    return n_errors
import numpy as np


def summarize_results(results, dataset, tol_obj=1e-4, tol_nmi=0):
    """All stored objectives must be minimized"""
    dcbm_algos = ("KL_EM", "KN", "KL_EM_SVCA", "KN_SVCA")
    groups = [dcbm_algos] + [(algo,) for algo in results if algo not in dcbm_algos]
    best = {}

    # Reference commune pour KN/KL-EM, propre a chaque autre methode
    for group in groups:
        candidates = [
            (algo, i)
            for algo in group if algo in results
            for i, obj in enumerate(results[algo]["obj_value"])
            if np.isfinite(obj)
        ]
        if candidates:
            reference = min(candidates, key=lambda x: results[x[0]]["obj_value"][x[1]])
            best.update({algo: reference for algo in group})

    summary = {}
    with open(f"{dataset}_results.txt", "w", encoding="utf-8") as file:
        for algo, data in results.items():
            objs = np.asarray(data["obj_value"], dtype=float)
            nmi = np.asarray(data["NMI"], dtype=float)
            best_obj = best_nmi = best_rmi = best_mis = np.nan
            n_opt, reached = None, "N/A"

            if algo in best:
                source, i = best[algo]
                ref = results[source]
                best_obj, best_nmi = ref["obj_value"][i], ref["NMI"][i]
                best_rmi, best_mis = ref["RMI"][i], ref["mis_nodes"][i]

                # Les DEUX conditions doivent etre satisfaites
                is_opt = ((np.abs(objs - best_obj)/np.abs(best_obj) <= tol_obj)
                          & (np.abs(nmi - best_nmi) <= tol_nmi))
                n_opt = int(np.sum(is_opt))
                reached = f"{n_opt}/{len(nmi)}"

            stats = {
                key: (np.mean(data[key]), np.std(data[key], ddof=1))
                for key in ("RMI", "NMI", "mis_nodes", "Time")
            }
            summary[algo] = dict(best_obj=best_obj, best_nmi=best_nmi,
                                 best_rmi=best_rmi, best_mis=best_mis,
                                 n_opt=n_opt, n_runs=len(nmi), stats=stats)
            line = (
                f"Algorithm: {algo}, "
                f"RMI (optimal): {best_rmi}, "
                f"RMI Mean: {stats['RMI'][0]:.4f}, "
                f"RMI Std: {stats['RMI'][1]:.4f}, "
                f"Misclassified Mean: {stats['mis_nodes'][0]:.4f}, "
                f"Misclassified Std: {stats['mis_nodes'][1]:.4f}, "
                f"Optimal obj: {best_obj:.6f}, "
                f"Misclassified (optimal): {best_mis}, "
                f"NMI (optimal): {best_nmi}, "
                f"Optimal reached: {reached}, "
                f"NMI Mean: {stats['NMI'][0]:.4f}, "
                f"NMI Std: {stats['NMI'][1]:.4f}, "
                f"Time Mean: {stats['Time'][0]:.4f}, "
                f"Time Std: {stats['Time'][1]:.4f}"
            )
            print(line)
            file.write(line + "\n")

    return summary

def main(graph, labels,dataset):
    seed=12
    nbr_tests = 100
    r = len(np.unique(labels))
    print(r)
    trials = 1
    results = {
        "KL_EM": {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "KN": {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "FROST_SVCA":  {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "KL_EM_SVCA":  {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "KN_SVCA":  {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "graph-tool": {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []},
        "SCORE+": {"mis_nodes":[],"obj_value":[], "NMI": [], "RMI": [], "Time": []}


    }
    for itt in range(nbr_tests):

        if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
           print(f"Test completed: {itt / nbr_tests * 100:.0f}%")


        seed=seed+itt
        # KL_EM
        start_time = time.time()
        EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
                            init_method="random", verbosity=0, init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        results["KL_EM"]["obj_value"].append(-obj)
        results["KL_EM"]["mis_nodes"].append(align_and_count_errors(labels,EM_partition))
        results["KL_EM"]["NMI"].append(NMI)
        results["KL_EM"]["RMI"].append(cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
        results["KL_EM"]["Time"].append(end_time-start_time)
        
        # KN
        start_time = time.time()
        EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=trials,
                            init_method="random", verbosity=0,init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        results["KN"]["obj_value"].append(-obj)
        results["KN"]["mis_nodes"].append(align_and_count_errors(labels,EM_partition))
        results["KN"]["NMI"].append(NMI)
        results["KN"]["RMI"].append(cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
        results["KN"]["Time"].append(end_time-start_time)

        # KL_EM initialized by SVCA
        start_time = time.time()
        EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
                            init_method="SVCA", verbosity=0, init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        results["KL_EM_SVCA"]["obj_value"].append(-obj)
        results["KL_EM_SVCA"]["mis_nodes"].append(align_and_count_errors(labels,EM_partition))
        results["KL_EM_SVCA"]["NMI"].append(NMI)
        results["KL_EM_SVCA"]["RMI"].append(cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
        results["KL_EM_SVCA"]["Time"].append(end_time-start_time)

        # KN initialized by SVCA
        start_time = time.time()
        EM_partition, obj = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=trials,
                            init_method="SVCA", verbosity=0, init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, EM_partition)
        results["KN_SVCA"]["obj_value"].append(-obj)
        results["KN_SVCA"]["mis_nodes"].append(align_and_count_errors(labels,EM_partition))
        results["KN_SVCA"]["NMI"].append(NMI)
        results["KN_SVCA"]["RMI"].append(cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
        results["KN_SVCA"]["Time"].append(end_time-start_time)

       
        # FROST initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA", numTrials=trials, verbosity=0, init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, v_best)
        results["FROST_SVCA"]["obj_value"].append(error_best)
        results["FROST_SVCA"]["mis_nodes"].append(align_and_count_errors(labels,v_best))
        results["FROST_SVCA"]["NMI"].append(NMI)
        results["FROST_SVCA"]["RMI"].append(cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
        results["FROST_SVCA"]["Time"].append(end_time - start_time)
        

        # Score+
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        SCORE_partition=SCORE_plus(X,r,init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, SCORE_partition)
        results["SCORE+"]["mis_nodes"].append(align_and_count_errors(labels,SCORE_partition))
        results["SCORE+"]["NMI"].append(NMI)
        results["SCORE+"]["RMI"].append(cmi.normalized_mutual_information(labels, SCORE_partition, variation="reduced", normalization="first"))
        results["SCORE+"]["Time"].append(end_time - start_time)

        # graphtool
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        graph_tool_partition, obj=graph_tool_dcbm(X,r,nTrials=trials,init_seed=seed)
        end_time = time.time()
        NMI = normalized_mutual_info_score(labels, graph_tool_partition)
        results["graph-tool"]["obj_value"].append(obj)
        results["graph-tool"]["mis_nodes"].append(align_and_count_errors(labels,graph_tool_partition))
        results["graph-tool"]["NMI"].append(NMI)
        results["graph-tool"]["RMI"].append(cmi.normalized_mutual_information(labels, graph_tool_partition, variation="reduced", normalization="first"))
        results["graph-tool"]["Time"].append(end_time - start_time)
       
        
    summary = summarize_results(
        results,
        dataset,
    )

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    dataset="caltech"

    if dataset=="karate":
        graph,labels=load_karate()
    elif dataset=="polblogs":
        graph,labels=load_polblogs()
    elif dataset=="football":
        graph,labels=load_football()
    elif dataset=="caltech":
        graph,labels=load_caltech()

    main(graph, labels, dataset)
