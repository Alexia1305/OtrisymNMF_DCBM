import numpy as np
import pysbm
import networkx as nx
import matplotlib.pyplot as plt
from Utils import DC_BM
import OtrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random

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
    results = {
        "OtrisymNMF": {"NMI": []},
        "KL_EM": {"NMI": []},
        "OtrisymNMF_SVCA": {"NMI": []},
        "KL_EM_SVCA": {"NMI": []},
        "SVCA": {"NMI": []}

    }
    for itt in range(nbr_tests):

        if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
            print(f"Test completed: {itt / nbr_tests * 100:.0f}%")



        # KL_EM
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
                             init_method="random", tri=False)
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM"]["NMI"].append(NMI)

        # OtrisymNMF
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="random", numTrials=10, verbosity=0)
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF"]["NMI"].append(NMI)

        # KL_EM initialized by SVCA

        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                             numTrials=10, init_method="SVCA", tri=False)
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM_SVCA"]["NMI"].append(NMI)

        # OtrisymNMF initialized by SVCA
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="SVCA", numTrials=10,verbosity=0)
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_SVCA"]["NMI"].append(NMI)

        # SVCA
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=50, verbosity=0)
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["SVCA"]["NMI"].append(NMI)

    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)}, ")

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
    graph, labels = read_graph()
    main(graph, labels)
