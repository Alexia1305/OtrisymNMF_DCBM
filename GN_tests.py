
import random
import matplotlib.pyplot as plt
import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from Utils import DC_BM
import csv

def Generate_GN(z, z_out):
    n = 128
    r = 4
    s = n // r
    z_in = z - z_out
    P_in = z_in / (s - 1)
    P_out = z_out / (n - s)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    assert n % r == 0, "n doit être divisible par r pour des groupes de taille égale."
    shuffle_indices = np.random.permutation(n)
    groups = [list(shuffle_indices[i * s: (i + 1) * s]) for i in range(r)]
    for group, community in zip(groups, range(r)):
        for node in group:
            G.nodes[node]['community'] = community

    for i in range(n):
        for j in range(n):
            if j < i:
                if G.nodes[i]['community'] == G.nodes[j]['community'] and random.random() < P_in:
                    G.add_edge(i, j)
                if G.nodes[i]['community'] != G.nodes[j]['community'] and random.random() < P_out:
                    G.add_edge(i, j)
    return G
def GN_test():
    z=16
    nbr_tests=100
    num_trials=10
    results_list = []  # Liste pour stocker les résultats
    for z_out in range(0,z+1):
        results = {
            "OtrisymNMF": {"NMI": [], "Time": []},
            "KL_G": {"NMI": [], "Time": []},
            "KL_EM": {"NMI": [], "Time": []},
            "MHA": {"NMI": [], "Time": []},
            "OtrisymNMF_SVCA": {"NMI": [], "Time": []},
            "KL_G_SVCA": {"NMI": [], "Time": []},
            "KL_EM_SVCA": {"NMI": [], "Time": []},
            "MHA_SVCA": {"NMI": [], "Time": []},
            "SVCA": {"NMI": [], "Time": []}
        }
        for test in range(nbr_tests):
            G=Generate_GN(z,z_out)
            labels = [G.nodes[v]['community'] for v in sorted(G.nodes)]
            r=max(labels)+1

            # OtrisymNMF random

            X = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))
            start_time = time.time()
            w_best, v_best, S_best, error_best, _ = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=num_trials,
                                                                             init_method="random", verbosity=0,
                                                                             init_seed=test)
            end_time = time.time()
            init_time = end_time - start_time
            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF"]["NMI"].append(NMI)
            results["OtrisymNMF"]["Time"].append(end_time - start_time)
            #print(f" Random Otri :{NMI}")

            # # KL_G
            start_time=time.time()
            KLG_partition=DC_BM(G, r,pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KarrerInference, numTrials=num_trials,verbosity=0,init_method="random")
            end_time=time.time()
            NMI=normalized_mutual_info_score(labels,KLG_partition)
            results["KL_G"]["NMI"].append(NMI)
            results["KL_G"]["Time"].append(end_time-start_time)
            #print(f" Random KN :{NMI}")
            # #
            # KL_EM
            start_time = time.time()
            EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=num_trials,verbosity=0,init_method="random")
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, EM_partition)
            results["KL_EM"]["NMI"].append(NMI)
            results["KL_EM"]["Time"].append(end_time-start_time)
            #print(f" Random KL_EM :{NMI}")
            # #
            # MHA10
            start_time = time.time()
            MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.MetropolisHastingInferenceTenK, numTrials=num_trials,init_method="random",verbosity=0)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, MHA_partition)
            results["MHA"]["NMI"].append(NMI)
            results["MHA"]["Time"].append(end_time - start_time)
            #print(f" Random MH :{NMI}")

            #OtrisymNMF

            X = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))
            start_time = time.time()
            w_best, v_best, S_best, error_best,_ = OtrisymNMF.OtrisymNMF_CD(X, r,numTrials=num_trials,init_method="SVCA",verbosity=0,init_seed=test)
            end_time = time.time()
            init_time=end_time - start_time
            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
            results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)
            #print(f" SVCA Otri :{NMI}")

            # MHA10 SVCA
            start_time = time.time()
            MHA_partition= DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
                                     pysbm.MetropolisHastingInferenceTenK, numTrials=num_trials, init_method="SVCA",verbosity=0,init_seed=test)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, MHA_partition)
            results["MHA_SVCA"]["NMI"].append(NMI)
            results["MHA_SVCA"]["Time"].append(end_time - start_time)
            #print(f" SVCA MH :{NMI}")

            #KL_EM SVCA

            start_time = time.time()
            EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=num_trials,init_method="SVCA",verbosity=0,init_seed=test)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, EM_partition)
            results["KL_EM_SVCA"]["NMI"].append(NMI)
            results["KL_EM_SVCA"]["Time"].append(end_time-start_time+ init_time)
            #print(f" SVCA KL_EM :{NMI}")

            # KN SVCA

            start_time = time.time()
            KN_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=num_trials, init_method="SVCA", verbosity=0,init_seed=test)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, KN_partition)
            results["KL_G_SVCA"]["NMI"].append(NMI)
            results["KL_G_SVCA"]["Time"].append(end_time - start_time + init_time)
            #print(f" SVCA KN :{NMI}")

            #SVCA
            start_time = time.time()
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            w_best, v, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=num_trials, verbosity=0, init_seed=test)
            NMI = normalized_mutual_info_score(labels, v)
            end_time = time.time()
            results["SVCA"]["NMI"].append(NMI)
            results["SVCA"]["Time"].append(end_time - start_time)
            #print(f"SVCA :{NMI}")

        # Calcul des statistiques
        for algo, data in results.items():
            results_list.append({
                "Algorithm": algo,
                "z_out": z_out,
                "NMI Mean": np.mean(data["NMI"]),
                "NMI Std": np.std(data["NMI"], ddof=1),
                "Time Mean (s)": np.mean(data["Time"]),
                "Time Std (s)": np.std(data["Time"], ddof=1)
            })
            print(
                f"Algorithm: {algo}, z_out: {z_out}, NMI Mean: {np.mean(data['NMI'])}, NMI Std: {np.std(data['NMI'], ddof=1)}, Time Mean (s): {np.mean(data['Time'])}, Time Std (s): {np.std(data['Time'], ddof=1)}")

    with open("results/GN_test/results.csv", "w", newline='') as csvfile:
        fieldnames = ["Algorithm", "z_out", "NMI Mean", "NMI Std", "Time Mean (s)", "Time Std (s)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
        # Sauvegarde dans un fichier CSV




if __name__ == "__main__":
    random.seed(42)
    GN_test()
