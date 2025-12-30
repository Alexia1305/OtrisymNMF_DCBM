import numpy as np
import pysbm
import networkx as nx
import matplotlib.pyplot as plt
from dcbm import dcbm
import otrisymNMF
from sklearn.metrics import normalized_mutual_info_score
import random, time

def read_graph():
    G = nx.read_gml('Data\polblogs_cleaned.gml', label='id')
    True_partition = [int(G.nodes[node]['value']) for node in G.nodes]

    return G, True_partition


def main(graph, clusters):
    nbr_tests = 100
    r = 2
    trials = 1
    results = {
        "FROST": {"NMI": [], "Success_rate": [], "Time": []},
        "KL_EM": {"NMI": [], "Success_rate": [] ,"Time": []},
        "KN": {"NMI": [], "Success_rate": [], "Time": []},
        "MH": {"NMI": [], "Success_rate": [], "Time": []},
        "FROST_SVCA": {"NMI": [], "Success_rate": [], "Time": []},
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
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=trials,
                            init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM"]["NMI"].append(NMI)
        results["KL_EM"]["Time"].append(end_time-start_time)
        if NMI >= 0.72867696256281:
            results["KL_EM"]["Success_rate"].append(1)


        # KN
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=trials,
                            init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN"]["NMI"].append(NMI)
        results["KN"]["Time"].append(end_time - start_time)
        if NMI >= 0.72867696256281:
            results["KN"]["Success_rate"].append(1)

        # MH
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK, numTrials=trials,
                            init_method="random", verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH"]["NMI"].append(NMI)
        results["MH"]["Time"].append(end_time - start_time)
        if NMI >= 0.72867696256281:
            results["MH"]["Success_rate"].append(1)


        # FROST
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best, time_per_iteration= otrisymNMF.frost(X, r, init_method="random", numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["FROST"]["NMI"].append(NMI)
        results["FROST"]["Time"].append(end_time - start_time)
        if NMI >= 0.722389545475209:
            results["FROST"]["Success_rate"].append(1)



        # KL_EM initialized by SVCA
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference,
                            numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KL_EM_SVCA"]["NMI"].append(NMI)
        results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
        if NMI >= 0.72867696256281:
            results["KL_EM_SVCA"]["Success_rate"].append(1)

        # KN initialized by SVCA
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                            numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["KN_SVCA"]["NMI"].append(NMI)
        results["KN_SVCA"]["Time"].append(end_time - start_time)
        if NMI >= 0.72867696256281:
            results["KN_SVCA"]["Success_rate"].append(1)
        #
        # MH initialized by SVCA
        start_time = time.time()
        EM_partition = dcbm(graph, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceHundredK,
                            numTrials=trials, init_method="SVCA", verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, EM_partition)
        results["MH_SVCA"]["NMI"].append(NMI)
        results["MH_SVCA"]["Time"].append(end_time - start_time)
        if NMI >= 0.72867696256281:
            results["MH_SVCA"]["Success_rate"].append(1)

        # FROST initialized by SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, init_method="SVCA", numTrials=trials, verbosity=0, init_seed=itt)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["FROST_SVCA"]["NMI"].append(NMI)
        results["FROST_SVCA"]["Time"].append(end_time - start_time)
        if NMI >=0.722389545475209:
            results["FROST_SVCA"]["Success_rate"].append(1)


        # SVCA
        start_time = time.time()
        X = nx.adjacency_matrix(graph)
        w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=trials, verbosity=0)
        end_time = time.time()
        NMI = normalized_mutual_info_score(clusters, v_best)
        results["SVCA"]["NMI"].append(NMI)
        results["SVCA"]["Time"].append(end_time - start_time)
        if NMI >=0.722389545475209:
            results["SVCA"]["Success_rate"].append(1)
    for algo, data in results.items():
        print(
            f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,Success rate {np.sum(data['Success_rate'])/nbr_tests}")

    with open('polblogs_results.txt', 'w') as file:
        for algo, data in results.items():
            # Calcul des statistiques
            nmi_mean = np.mean(data['NMI'])
            nmi_std = np.std(data['NMI'], ddof=1)

            # Enregistrer les résultats dans le fichier texte
            file.write(f"Algorithm: {algo}, NMI Mean: {np.round(np.mean(data['NMI']),4)}, NMI Std: {np.round(np.std(data['NMI'], ddof=1),4)},Time Mean: {np.round(np.mean(data['Time']),4)}, Time Std: {np.round(np.std(data['Time'], ddof=1),4)} ,Success rate {np.sum(data['Success_rate'])/nbr_tests}\n")


if __name__ == "__main__":
    random.seed(12)
    np.random.seed(40)
    graph, labels = read_graph()
    main(graph, labels)
