import numpy as np
import pysbm
import networkx as nx
import matplotlib.pyplot as plt
from Utils import DC_BM
import OtrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time

def read_graph():
    file_path = "Data/Scotland.net"
    G = nx.read_pajek(file_path)
    G = nx.Graph(G)
    clusters = G.nodes

    # node 0 to 107 are compagnies and 108 to 243 are administrator
    clusters = np.ones(len(clusters), dtype=int)
    clusters[:108] = 0

    # Removal of isolated nodes and nodes disconnected from the largest component
    isolated_nodes = list(nx.isolates(G))
    isolated_nodes = isolated_nodes + [list(G.nodes)[i] for i in [138, 71, 137, 4, 170, 33, 106, 31, 157, 12, 158]]
    node_indices = [list(G.nodes).index(node) for node in isolated_nodes]
    G.remove_nodes_from(isolated_nodes)
    clusters = np.delete(clusters, node_indices)

    # Network display
    node_colors = ['red' if label == 1 else 'blue' for label in clusters]
    plt.figure(figsize=(10, 7))
    nx.draw(G, nx.circular_layout(G), with_labels=False, node_color=node_colors, node_size=30, font_size=12,
            width=0.5)
    plt.show()

    return G, clusters


def main(graph, clusters):
    nbr_tests = 100
    r = 2
    runs = 1
    results = {
        "OtrisymNMF": {"NMI": [], "Success_rate": [], "Time": []},
        "KL_EM": {"NMI": [], "Success_rate": [] ,"Time": []},
        "KN": {"NMI": [], "Success_rate": [], "Time": []},
        "MH": {"NMI": [], "Success_rate": [], "Time": []},
        "OtrisymNMF_SVCA": {"NMI": [], "Success_rate": [], "Time": []},
        "KL_EM_SVCA": {"NMI": [], "Success_rate": [], "Time": []},
        "KN_SVCA": {"NMI": [], "Success_rate": [], "Time": []},
        "MH_SVCA": {"NMI": [], "Success_rate": [], "Time": []},
        "SVCA": {"NMI": [], "Success_rate": [],  "Time": []}

    }
    for itt in range(nbr_tests):

        if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
           print(f"Test completed: {itt / nbr_tests * 100:.0f}%")



        # KL_EM
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=runs,
                             init_method="random", tri=False, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM"]["NMI"].append(NMI)
        results["KL_EM"]["Time"].append(end_time-start_time)
        if NMI == 1:
            results["KL_EM"]["Success_rate"].append(1)


        # KN
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=runs,
                             init_method="random", tri=False, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN"]["NMI"].append(NMI)
        results["KN"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["KN"]["Success_rate"].append(1)

        # MH
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceFiftyK, numTrials=runs,
                             init_method="random", tri=False, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH"]["NMI"].append(NMI)
        results["MH"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["MH"]["Success_rate"].append(1)


        # OtrisymNMF
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="random", numTrials=runs, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF"]["NMI"].append(NMI)
        results["OtrisymNMF"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["OtrisymNMF"]["Success_rate"].append(1)

        # KL_EM initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                             numTrials=runs, init_method="SVCA", tri=False, verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM_SVCA"]["NMI"].append(NMI)
        results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["KL_EM_SVCA"]["Success_rate"].append(1)

        # KN initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                             numTrials=runs, init_method="SVCA", tri=False, verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN_SVCA"]["NMI"].append(NMI)
        results["KN_SVCA"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["KN_SVCA"]["Success_rate"].append(1)

        # MH initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceFiftyK,
                             numTrials=runs, init_method="SVCA", tri=False, verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH_SVCA"]["NMI"].append(NMI)
        results["MH_SVCA"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["MH_SVCA"]["Success_rate"].append(1)

        # OtrisymNMF initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="SVCA", numTrials=runs,verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
        results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["OtrisymNMF_SVCA"]["Success_rate"].append(1)
        # SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=runs, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["SVCA"]["NMI"].append(NMI)
        results["SVCA"]["Time"].append(end_time - start_time)
        if NMI == 1:
            results["SVCA"]["Success_rate"].append(1)
    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,Success rate {np.sum(data['Success_rate'])/nbr_tests}")

    # with open('Scotland.txt', 'w') as file:
    #     for algo, data in results.items():
    #         # Calcul des statistiques
    #         nmi_mean = np.mean(data['NMI'])
    #         nmi_std = np.std(data['NMI'], ddof=1)
    #
    #         # Enregistrer les résultats dans le fichier texte
    #         file.write(f"Algorithm: {algo}, NMI Mean: {nmi_mean}, NMI Std: {nmi_std}\n")


if __name__ == "__main__":
    random.seed(15)  # Fixer la seed
    np.random.seed(125)
    graph, labels = read_graph()
    main(graph, labels)
