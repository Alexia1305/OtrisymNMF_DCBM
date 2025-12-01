import numpy as np
import pysbm
import networkx as nx
import matplotlib.pyplot as plt
from Utils import DC_BM
import OtrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time

def read_graph():

    # plt.rcParams['text.usetex'] = True  # Active le rendu LaTeX
    G = nx.davis_southern_women_graph()  # Get the Southern Women network
    # Recovery of the women and events from node's graph
    types = [d['bipartite'] for n,d in G.nodes(data=True)]
    women = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    events = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]

    # Network Display
    pos = nx.spring_layout(G, seed=12)

    node_list = list(G.nodes())
    plt.figure(figsize=(15, 12))

    nx.draw_networkx_nodes(G, pos,
                           nodelist=women,
                           node_shape='o',
                           node_color='gray',
                           node_size=600)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=events,
                           node_color="gray",
                           node_shape='s',
                           node_size=600)

    nx.draw_networkx_edges(G, pos, edge_color='gray')
    # labels
    labels = {}
    for n in G.nodes():
        if n in women:
            parts = n.split()
            initials = '. '.join(p[0] for p in parts) + '.'
            labels[n] = initials
        else:
            labels[n] = n  # garder les événements inchangés

    # Affichage des labels personnalisés
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=18)

    plt.title("Southern Women Network", fontsize=18)
    plt.axis('off')
    plt.show()

    return G, types


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



        # # KL_EM
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=runs,
        #                      init_method="random", verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["KL_EM"]["NMI"].append(NMI)
        # results["KL_EM"]["Time"].append(end_time-start_time)
        # if NMI == 1:
        #     results["KL_EM"]["Success_rate"].append(1)
        #
        #
        # # KN
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=runs,
        #                      init_method="random", verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["KN"]["NMI"].append(NMI)
        # results["KN"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["KN"]["Success_rate"].append(1)
        #
        # # MH
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceTenK, numTrials=runs,
        #                      init_method="random", verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["MH"]["NMI"].append(NMI)
        # results["MH"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["MH"]["Success_rate"].append(1)
        #
        #
        # # OtrisymNMF
        # start_time = time.time()
        # X = nx.adjacency_matrix(graph)
        # w_best, v_best, S_best, error_best,time_it = OtrisymNMF.OtrisymNMF_CD(X, r,init_method="random", numTrials=runs, verbosity=0,init_seed=itt,delta=1e-5)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, v_best)
        # results["OtrisymNMF"]["NMI"].append(NMI)
        # results["OtrisymNMF"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["OtrisymNMF"]["Success_rate"].append(1)
        #
        #
        #
        # # KL_EM initialized by SVCA
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
        #                      numTrials=runs, init_method="SVCA", verbosity=0, init_seed=itt)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["KL_EM_SVCA"]["NMI"].append(NMI)
        # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["KL_EM_SVCA"]["Success_rate"].append(1)
        #
        # # KN initialized by SVCA
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
        #                      numTrials=runs, init_method="SVCA", verbosity=0, init_seed=itt)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["KN_SVCA"]["NMI"].append(NMI)
        # results["KN_SVCA"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["KN_SVCA"]["Success_rate"].append(1)
        #
        # # MH initialized by SVCA
        # start_time = time.time()
        # EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceTenK,
        #                      numTrials=runs, init_method="SVCA", verbosity=0, init_seed=itt)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, EM_partition)
        # results["MH_SVCA"]["NMI"].append(NMI)
        # results["MH_SVCA"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["MH_SVCA"]["Success_rate"].append(1)

        # OtrisymNMF initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best,time_it = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="SVCA", numTrials=runs,verbosity=1, init_seed=itt,delta=1e-5)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
        results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)
        print(NMI)
        if NMI == 1:
            results["OtrisymNMF_SVCA"]["Success_rate"].append(1)

        # # SVCA
        # start_time = time.time()
        # X = nx.adjacency_matrix(graph)
        # w_best, v_best, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=runs, verbosity=0)
        # end_time = time.time()
        # NMI = normalized_mutual_info_score(clusters, v_best)
        # results["SVCA"]["NMI"].append(NMI)
        # results["SVCA"]["Time"].append(end_time - start_time)
        # if NMI == 1:
        #     results["SVCA"]["Success_rate"].append(1)
    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,Success rate {np.sum(data['Success_rate'])/nbr_tests}")

    with open('Southern_results.txt', 'w') as file:
        for algo, data in results.items():
            # Calcul des statistiques
            nmi_mean = np.mean(data['NMI'])
            nmi_std = np.std(data['NMI'], ddof=1)

            # Enregistrer les résultats dans le fichier texte
            #file.write(f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,Success rate {np.sum(data['Success_rate'])/nbr_tests} \n")



if __name__ == "__main__":
    random.seed(25)  # Fixer la seed
    np.random.seed(125)
    graph, labels = read_graph()
    main(graph, labels)
