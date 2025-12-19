import json

import otrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from dcbm import dcbm
import random
from scipy.io import savemat


def read_graphs_from_files(graphs_folder, n):
    """Reading the LFR benchmark graph data"""
    graphs = []

    for i in range(1, 11):

        network_file = f"{graphs_folder}/network_{i}.dat"
        graph = nx.Graph()
        graph.add_nodes_from(range(1, n + 1))
        with open(network_file, 'r') as file:
            for line in file:
                u, v = map(int, line.split())
                graph.add_edge(u, v)

        community_file = f"{graphs_folder}/community_{i}.dat"

        with open(community_file, 'r') as file:
            for line in file:
                parts = line.split()
                node = int(parts[0])
                community = int(parts[1])
                graph.nodes[node]['community'] = community

        graphs.append(graph)

    return graphs


def main(list_n):
    """ Test LFR benchmark """

    numTrials = 1
    Time_limit = 1000000

    for n in list_n:
        graphs_folder = f"Data/LFR_N/n_{n}"
        graphs = read_graphs_from_files(graphs_folder, n)
        results = {
            "KN": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "KL_EM": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "FROST": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "KN_SVCA": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "KL_EM_SVCA": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "FROST_SVCA": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []},
            "SVCA": {"NMI": [], "AMI": [], "Time": [], "Nbr_iterations": [], "Time_iterations": []}

        }
        r_liste = []
        for idx, G in enumerate(graphs, start=1):

            print(f"Processed {idx} out of {len(graphs)} graphs.")

            labels = [G.nodes[v]['community'] for v in G.nodes]
            r = max(labels)
            r_liste.append(r)

            # SVCA
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=numTrials, init_seed=idx, verbosity=0)
            end_time = time.time()
            AMI = adjusted_mutual_info_score(labels, v_best)
            NMI = normalized_mutual_info_score(labels, v_best)
            results["SVCA"]["NMI"].append(NMI)
            results["SVCA"]["AMI"].append(AMI)
            results["SVCA"]["Time"].append(end_time - start_time)
            results["SVCA"]["Nbr_iterations"].append(0)
            results["SVCA"]["Time_iterations"].append(0)


            # FROST
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best,time_per_iteration = otrisymNMF.frost(X, r, numTrials=numTrials, init_method="random",
                                                                                             time_limit=Time_limit, init_seed=idx,
                                                                                             delta=1e-5, verbosity=0)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            AMI = adjusted_mutual_info_score(labels, v_best)
            results["FROST"]["NMI"].append(NMI)
            results["FROST"]["AMI"].append(AMI)
            results["FROST"]["Time"].append(end_time - start_time)
            results["FROST"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["FROST"]["Time_iterations"].extend(time_per_iteration[0])



            # #KN
            # start_time=time.time()
            # KLG_partition,time_per_iteration=dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KarrerInference, numTrials=numTrials,
            #                     init_method="random", verbosity=0, time_limit=Time_limit,need_time=True)
            # end_time=time.time()
            # NMI = normalized_mutual_info_score(labels,KLG_partition)
            # AMI = adjusted_mutual_info_score(labels, KLG_partition)
            # results["KN"]["NMI"].append(NMI)
            # results["KN"]["AMI"].append(AMI)
            # results["KN"]["Time"].append(end_time-start_time)
            # results["KN"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            # results["KN"]["Time_iterations"].extend(time_per_iteration[0])


            # # KL_EM
            # start_time = time.time()
            # EM_partition,time_per_iteration = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=numTrials,
            #                      init_method="random", verbosity=0, time_limit=Time_limit,need_time=True)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, EM_partition)
            # AMI = adjusted_mutual_info_score(labels, EM_partition)
            # results["KL_EM"]["NMI"].append(NMI)
            # results["KL_EM"]["AMI"].append(AMI)
            # results["KL_EM"]["Time"].append(end_time - start_time)
            # results["KL_EM"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            # results["KL_EM"]["Time_iterations"].extend(time_per_iteration[0])



            #FROST SVCA
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best, time_per_iteration = otrisymNMF.frost(X, r, numTrials=numTrials, init_method="SVCA", time_limit=Time_limit, init_seed=idx, delta=1e-5, verbosity=0)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            AMI = adjusted_mutual_info_score(labels, v_best)
            results["FROST_SVCA"]["NMI"].append(NMI)
            results["FROST_SVCA"]["AMI"].append(AMI)
            results["FROST_SVCA"]["Time"].append(end_time - start_time)
            results["FROST_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["FROST_SVCA"]["Time_iterations"].extend(time_per_iteration[0])



            # #KN initialized by SVCA
            # start_time = time.time()
            # KLG_partition,time_per_iteration = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
            #                       numTrials=numTrials, init_method="SVCA", verbosity=0, init_seed=idx, time_limit=Time_limit,need_time=True)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, KLG_partition)
            # AMI = adjusted_mutual_info_score(labels, KLG_partition)
            # results["KN_SVCA"]["NMI"].append(NMI)
            # results["KN_SVCA"]["AMI"].append(AMI)
            # results["KN_SVCA"]["Time"].append(end_time - start_time)
            # results["KN_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            # results["KN_SVCA"]["Time_iterations"].extend(time_per_iteration[0])
            #
            #
            # # KL_EM initialized by SVCA
            # start_time = time.time()
            # EM_partition,time_per_iteration = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=numTrials,
            #                      init_method="SVCA", verbosity=0, init_seed=idx, time_limit=Time_limit,need_time=True)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, EM_partition)
            # AMI = adjusted_mutual_info_score(labels, EM_partition)
            # results["KL_EM_SVCA"]["NMI"].append(NMI)
            # results["KL_EM_SVCA"]["AMI"].append(AMI)
            # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            # results["KL_EM_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            # results["KL_EM_SVCA"]["Time_iterations"].extend(time_per_iteration[0])



        summary = {}
        for algo, data in results.items():
            summary[algo] = {
                "Mean NMI": np.round(np.mean(data["NMI"]), 3),
                "SD NMI": np.round(np.std(data["NMI"], ddof=1), 3),
                "Mean AMI": np.round(np.mean(data["AMI"]), 3),
                "SD AMI": np.round(np.std(data["AMI"], ddof=1), 3),
                "Mean Time (s)": np.round(np.mean(data["Time"]), 2),
                "SD Time (s)": np.round(np.std(data["Time"], ddof=1), 2),
                "Mean Nbr of iterations": np.round(np.mean(data["Nbr_iterations"]),2),
                "SD Nbr of iterations": np.round(np.std(data["Nbr_iterations"],ddof=1),2),
                "Mean Time of one iteration": np.round(np.mean(data["Time_iterations"]),2),
                "SD Time of one iteration": np.round(np.std(data["Time_iterations"],ddof=1),2)

            }

        r_liste.sort()
        print(r_liste)

        # Results save
        r_liste_filename = f"n_{n:.1f}_r_liste.txt"
        with open(r_liste_filename, "w") as f:
            for item in r_liste:
                f.write(f"{item}\n")

        with open(f"results/Test_scalability/n_{n:.1f}_results_f.json", "w") as f:
            json.dump(results, f, indent=4)

        df_results = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\nRésultats pour n={n:.1f}:")
        # Results Display
        print(df_results)

        results_filename = f"results/Test_scalability/n_{n:.1f}_results_f.csv"
        df_results.to_csv(results_filename)
        print(f"Résultats enregistrés dans '{results_filename}'\n")


if __name__ == "__main__":

    #Options TEST
    list_n = [100000]

    random.seed(42)  # Fixer la seed
    main(list_n)

