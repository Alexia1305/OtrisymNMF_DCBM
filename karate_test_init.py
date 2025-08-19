import numpy as np
import pysbm
import networkx as nx
import matplotlib.pyplot as plt
from Utils import DC_BM
import OtrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time

def read_graph():
    G = nx.karate_club_graph()
    club_labels = {node: 1 if G.nodes[node]['club'] == 'Mr. Hi' else 0 for node in G.nodes}
    True_partition = np.array([club_labels[node] for node in G.nodes])

    return G, True_partition


def main(graph, clusters):
    nbr_tests = 100
    r = 2
    trials = 1
    part_one_node = np.array([1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
    part_two_nodes =np.array([1 ,1 ,1 , 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    results = {
        "OtrisymNMF": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "OtrisymNMF_S": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "KL_EM": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0 ,"Time": []},
        "KN": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "MH": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "OtrisymNMF_SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "OtrisymNMF_S_SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "KL_EM_SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "KN_SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "MH_SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0, "Time": []},
        "SVCA": {"NMI": [], "One_node_misc": 0, "Two_nodes_misc":0,  "Time": []}

    }
    for itt in range(nbr_tests):

        if itt % (nbr_tests // 10) == 0:  # Afficher tous les 10 %
            print(f"Test completed: {itt / nbr_tests * 100:.0f}%")



        # KL_EM
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
                             init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM"]["NMI"].append(NMI)
        results["KL_EM"]["Time"].append(end_time-start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["KL_EM"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["KL_EM"]["Two_nodes_misc"] += 1


        # KN
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=trials,
                             init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN"]["NMI"].append(NMI)
        results["KN"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["KN"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["KN"]["Two_nodes_misc"] += 1

        # MH
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK, numTrials=trials,
                             init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH"]["NMI"].append(NMI)
        results["MH"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["MH"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["MH"]["Two_nodes_misc"] += 1


        # OtrisymNMF
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="random", numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF"]["NMI"].append(NMI)
        results["OtrisymNMF"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, v_best) == 1:
            results["OtrisymNMF"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, v_best) == 1:
            results["OtrisymNMF"]["Two_nodes_misc"] += 1

        # OtrisymNMF_S
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="random", update_rule="S_direct",
                                                                      numTrials=trials,verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_S"]["NMI"].append(NMI)
        results["OtrisymNMF_S"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, v_best) == 1:
            results["OtrisymNMF_S"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, v_best) == 1:
            results["OtrisymNMF_S"]["Two_nodes_misc"] += 1

        # KL_EM initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                             numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM_SVCA"]["NMI"].append(NMI)
        results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["KL_EM_SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["KL_EM_SVCA"]["Two_nodes_misc"] += 1

        # KN initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                             numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN_SVCA"]["NMI"].append(NMI)
        results["KN_SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["KN_SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["KN_SVCA"]["Two_nodes_misc"] += 1

        # MH initialized by SVCA
        start_time = time.time()
        EM_partition = DC_BM(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK,
                             numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH_SVCA"]["NMI"].append(NMI)
        results["MH_SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, EM_partition) == 1:
            results["MH_SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, EM_partition) == 1:
            results["MH_SVCA"]["Two_nodes_misc"] += 1

        # OtrisymNMF initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="SVCA", numTrials=trials,verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
        results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, v_best) == 1:
            results["OtrisymNMF_SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, v_best) == 1:
            results["OtrisymNMF_SVCA"]["Two_nodes_misc"] += 1

        # OtrisymNMF_S initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, init_method="SVCA", numTrials=trials,
                                                                      verbosity=0, init_seed=itt,update_rule="S_direct")
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["OtrisymNMF_S_SVCA"]["NMI"].append(NMI)
        results["OtrisymNMF_S_SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, v_best) == 1:
            results["OtrisymNMF_S_SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, v_best) == 1:
            results["OtrisymNMF_S_SVCA"]["Two_nodes_misc"] += 1
        # SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["SVCA"]["NMI"].append(NMI)
        results["SVCA"]["Time"].append(end_time - start_time)
        if normalized_mutual_info_score(part_one_node, v_best) == 1:
            results["SVCA"]["One_node_misc"] += 1
        elif normalized_mutual_info_score(part_two_nodes, v_best) == 1:
            results["SVCA"]["Two_nodes_misc"] += 1
    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,One node misclassified {data['One_node_misc']/nbr_tests},Two nodes misclassified {data['Two_nodes_misc']/nbr_tests}")

    with open('Karate2.txt', 'w') as file:
        for algo, data in results.items():
            # Calcul des statistiques
            nmi_mean = np.mean(data['NMI'])
            nmi_std = np.std(data['NMI'], ddof=1)

            # Enregistrer les résultats dans le fichier texte
            file.write(f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,One node misclassified {data['One_node_misc']/nbr_tests},Two nodes misclassified {data['Two_nodes_misc']/nbr_tests}\n")


if __name__ == "__main__":
    random.seed(12)
    np.random.seed(40)
    graph, labels = read_graph()
    main(graph, labels)
