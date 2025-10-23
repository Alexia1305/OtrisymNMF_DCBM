
import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from Utils import DC_BM
import random
from scipy.io import savemat
import json
from sklearn.metrics import adjusted_mutual_info_score

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


def main(list_n,degree):
    """ Test LFR benchmark """

    numTrials = 10
    Time_limit = 10000
    n = 1000

    for mu in list_mu:
        graphs_folder = f"Data/LFR_D/d{degree}/mu_{mu}"
        graphs = read_graphs_from_files(graphs_folder, n)
        results = {
            "OtrisymNMF": {"NMI": [], "AMI":[], "Time": [], "Trials": [], "Nbr_iterations":[],"Time_iterations":[]},
            "OtrisymNMF_SVCA": {"NMI": [], "AMI":[], "Time": [],"Trials": [], "Nbr_iterations":[],"Time_iterations":[]},
            "KN": {"NMI": [],  "AMI":[],"Time": [], "Trials": [], "Nbr_iterations":[], "Time_iterations":[]},
            "KL_EM": {"NMI": [],  "AMI":[],"Time": [], "Trials": [], "Nbr_iterations":[], "Time_iterations":[]},
            "KN_SVCA": {"NMI": [], "AMI":[], "Time": [], "Trials": [], "Nbr_iterations":[], "Time_iterations":[]},
            "KL_EM_SVCA": {"NMI": [], "AMI":[], "Time": [], "Trials": [], "Nbr_iterations":[], "Time_iterations":[]},

        }
        r_liste = []
        for idx, G in enumerate(graphs, start=1):

            print(f"Processed {idx} out of {len(graphs)} graphs.")

            labels = [G.nodes[v]['community'] for v in G.nodes]
            r = max(labels)
            r_liste.append(r)


            # OtrisymNMF
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best,time_per_iteration = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=numTrials, init_method="random",
                                                                          time_limit=Time_limit, init_seed=idx,
                                                                          delta=1e-5, verbosity=0)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            AMI = adjusted_mutual_info_score(labels,v_best)
            results["OtrisymNMF"]["NMI"].append(NMI)
            results["OtrisymNMF"]["AMI"].append(AMI)
            results["OtrisymNMF"]["Time"].append(end_time - start_time)
            results["OtrisymNMF"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["OtrisymNMF"]["Time_iterations"].extend(time_per_iteration[0])
            results["OtrisymNMF"]["Trials"].append(len(time_per_iteration))


            #KN
            start_time=time.time()
            KLG_partition,time_per_iteration=DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KarrerInference, numTrials=numTrials,
                                init_method="random", verbosity=0, time_limit=Time_limit,need_time=True)
            end_time=time.time()
            NMI=normalized_mutual_info_score(labels,KLG_partition)
            AMI = adjusted_mutual_info_score(labels, KLG_partition)
            results["KN"]["NMI"].append(NMI)
            results["KN"]["AMI"].append(AMI)
            results["KN"]["Time"].append(end_time-start_time)
            results["KN"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["KN"]["Time_iterations"].extend(time_per_iteration[0])
            results["KN"]["Trials"].append(len(time_per_iteration))

            # KL_EM
            start_time = time.time()
            EM_partition,time_per_iteration = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=numTrials,
                                 init_method="random", verbosity=0, time_limit=Time_limit,need_time=True)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, EM_partition)
            AMI = adjusted_mutual_info_score(labels, EM_partition)
            results["KL_EM"]["NMI"].append(NMI)
            results["KL_EM"]["AMI"].append(AMI)
            results["KL_EM"]["Time"].append(end_time - start_time)
            results["KL_EM"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["KL_EM"]["Time_iterations"].extend(time_per_iteration[0])
            results["KL_EM"]["Trials"].append(len(time_per_iteration))


            #OtrisymNMF SVCA
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best, time_per_iteration = OtrisymNMF.OtrisymNMF_CD(X,r,numTrials=numTrials,init_method="SVCA",time_limit=Time_limit, init_seed=idx,delta=1e-5, verbosity=0)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            AMI = adjusted_mutual_info_score(labels, v_best)
            results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
            results["OtrisymNMF_SVCA"]["AMI"].append(AMI)
            results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)
            results["OtrisymNMF_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["OtrisymNMF_SVCA"]["Time_iterations"].extend(time_per_iteration[0])
            results["OtrisymNMF_SVCA"]["Trials"].append(len(time_per_iteration))



            #KN initialized by SVCA
            start_time = time.time()
            KLG_partition,time_per_iteration = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                  numTrials=numTrials, init_method="SVCA", verbosity=0, init_seed=idx, time_limit=Time_limit,need_time=True)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, KLG_partition)
            AMI = adjusted_mutual_info_score(labels, KLG_partition)
            results["KN_SVCA"]["NMI"].append(NMI)
            results["KN_SVCA"]["AMI"].append(AMI)
            results["KN_SVCA"]["Time"].append(end_time - start_time)
            results["KN_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["KN_SVCA"]["Time_iterations"].extend(time_per_iteration[0])
            results["KN_SVCA"]["Trials"].append(len(time_per_iteration))

            # KL_EM initialized by SVCA
            start_time = time.time()
            EM_partition,time_per_iteration = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=numTrials,
                                 init_method="SVCA", verbosity=0, init_seed=idx, time_limit=Time_limit,need_time=True)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, EM_partition)
            AMI = adjusted_mutual_info_score(labels, EM_partition)
            results["KL_EM_SVCA"]["NMI"].append(NMI)
            results["KL_EM_SVCA"]["AMI"].append(AMI)
            results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            results["KL_EM_SVCA"]["Nbr_iterations"].append(len(time_per_iteration[0]))
            results["KL_EM_SVCA"]["Time_iterations"].extend(time_per_iteration[0])
            results["KL_EM_SVCA"]["Trials"].append(len(time_per_iteration))



        summary = {}
        for algo, data in results.items():
            summary[algo] = {
                "Mean NMI": np.round(np.mean(data["NMI"]), 4),
                "SD NMI": np.round(np.std(data["NMI"], ddof=1), 4),
                "Mean AMI": np.round(np.mean(data["AMI"]), 4),
                "SD AMI": np.round(np.std(data["AMI"], ddof=1), 4),
                "Mean Time (s)": np.round(np.mean(data["Time"]), 2),
                "SD Time (s)": np.round(np.std(data["Time"], ddof=1), 2),
                "Mean Nbr of trials": np.round(np.mean(data["Trials"]), 2),
                "SD Nbr of trials": np.round(np.std(data["Trials"], ddof=1), 2),
                "Mean Nbr of iterations" : np.round(np.mean(data["Nbr_iterations"]),2),
                "SD Nbr of iterations": np.round(np.std(data["Nbr_iterations"],ddof=1),2),
                "Mean Time of one iteration": np.round(np.mean(data["Time_iterations"]),2),
                "SD Time of one iteration":np.round(np.std(data["Time_iterations"],ddof=1),2)

            }

        with open(f"results/LFR_degree/mu_{mu:.1f}_results.json", "w") as f:
            json.dump(results, f, indent=4)


        df_results = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\nRésultats pour mu={mu:.1f}:")
        # Results Display
        print(df_results)

        results_filename = f"results/LFR_degree/mu_{mu:.1f}_results.csv"
        df_results.to_csv(results_filename)
        print(f"Résultats enregistrés dans '{results_filename}'\n")

        r_liste_filename = f"results/LFR_degree/mu_{mu:.1f}_r_liste.txt"
        with open(r_liste_filename, "w") as f:
            for item in r_liste:
                f.write(f"{item}\n")


if __name__ == "__main__":

    #Options TEST
    list_mu = [0.4]
    degree = 40
    random.seed(42)  # Fixer la seed
    main(list_mu,degree)

