
import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from Utils import DC_BM
import random


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


def main(list_mu):
    """ Test LFR benchmark """
    n = 1000
    for mu in list_mu:
        graphs_folder = f"Data/LFR/mu_{mu:.1f}"
        graphs = read_graphs_from_files(graphs_folder, n)
        results = {
            "OtrisymNMF": {"NMI": [], "Time": []},
            "OtrisymNMF_S": {"NMI": [], "Time": []},
            "KN": {"NMI": [], "Time": []},
            "KL_EM": {"NMI": [], "Time": []},
            "MHA250k": {"NMI": [], "Time": []},
            "OtrisymNMF_SVCA": {"NMI": [], "Time": []},
            "OtrisymNMF_S_SVCA": {"NMI": [], "Time": []},
            "SVCA": {"NMI": [], "Time": []},
            "KN_SVCA": {"NMI": [], "Time": []},
            "KL_EM_SVCA": {"NMI": [], "Time": []},
            "MHA250k_SVCA": {"NMI": [], "Time": []},

        }

        for idx, G in enumerate(graphs, start=1):

            print(f"Processed {idx} out of {len(graphs)} graphs.")

            labels = [G.nodes[v]['community'] for v in G.nodes]
            r = max(labels)

            # OtrisymNMF
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=10, init_method="random",
                                                                          verbosity=0, init_seed=idx, delta=1e-5)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF"]["NMI"].append(NMI)
            results["OtrisymNMF"]["Time"].append(end_time - start_time)

            # OtrisymNMF_S
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD_Sdirect(X, r, numTrials=10,
                                                                                  init_method="random",
                                                                                  verbosity=0, init_seed=idx,
                                                                                  delta=1e-5)
            end_time = time.time()

            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF_S"]["NMI"].append(NMI)
            results["OtrisymNMF_S"]["Time"].append(end_time - start_time)

            # # KN
            # start_time=time.time()
            # KLG_partition=DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KarrerInference, numTrials=10,
            #                     init_method="random", verbosity=0)
            # end_time=time.time()
            # NMI=normalized_mutual_info_score(labels,KLG_partition)
            # results["KN"]["NMI"].append(NMI)
            # results["KN"]["Time"].append(end_time-start_time)
            # #print(NMI)
            #
            # # KL_EM
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="random", verbosity=0)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, EM_partition)
            # results["KL_EM"]["NMI"].append(NMI)
            # results["KL_EM"]["Time"].append(end_time - start_time)
            # #print(NMI)
            #
            # # MHA250
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,init_method="random",
            #                       verbosity=0)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, MHA_partition)
            # results["MHA250k"]["NMI"].append(NMI)
            # results["MHA250k"]["Time"].append(end_time - start_time)
            # #print(NMI)

            #OtrisymNMF
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD(X, r,numTrials=10,init_method="SVCA",verbosity=0, init_seed=idx,delta=1e-5)
            end_time = time.time()
            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF_SVCA"]["NMI"].append(NMI)
            results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)



            # OtrisymNMF_S
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best = OtrisymNMF.OtrisymNMF_CD_Sdirect(X, r, numTrials=10, init_method="SVCA",
                                                                          verbosity=0, init_seed=idx, delta=1e-5)
            end_time = time.time()

            NMI = normalized_mutual_info_score(labels, v_best)
            results["OtrisymNMF_S_SVCA"]["NMI"].append(NMI)
            results["OtrisymNMF_S_SVCA"]["Time"].append(end_time - start_time)



            # #SVCA only
            # start_time = time.time()
            # X = nx.adjacency_matrix(G, nodelist=G.nodes)
            # w_best, v, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=10, verbosity=0)
            # NMI = normalized_mutual_info_score(labels, v)
            # end_time = time.time()
            # results["SVCA"]["NMI"].append(NMI)
            # results["SVCA"]["Time"].append(end_time - start_time)


            # # KN initialized by SVCA
            # start_time = time.time()
            # KLG_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
            #                       numTrials=10, init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, KLG_partition)
            # results["KN_SVCA"]["NMI"].append(NMI)
            # results["KN_SVCA"]["Time"].append(end_time - start_time)
            # #print(NMI)
            #
            # # KL_EM initialized by SVCA
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, EM_partition)
            # results["KL_EM_SVCA"]["NMI"].append(NMI)
            # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            # #print(NMI)
            #
            # # MHA250 initialized by SVCA
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,
            #                       init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # NMI = normalized_mutual_info_score(labels, MHA_partition)
            # results["MHA250k_SVCA"]["NMI"].append(NMI)
            # results["MHA250k_SVCA"]["Time"].append(end_time - start_time)
            # #print(NMI)

        summary = {}
        for algo, data in results.items():
            summary[algo] = {
                "NMI moyen": np.round(np.mean(data["NMI"]), 4),
                "Erreur type NMI": np.round(np.std(data["NMI"], ddof=1), 4),
                "Temps moyen (s)": np.round(np.mean(data["Time"]), 2),
                "Erreur type Temps": np.round(np.std(data["Time"], ddof=1), 2)
            }

        break
        df_results = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\nRésultats pour mu={mu:.1f}:")
        # Results Display
        print(df_results)

        # Sauvegarde des résultats dans un fichier CSV
        results_filename = f"mu_{mu:.1f}_resultsOtri.csv"
        df_results.to_csv(results_filename)
        print(f"Résultats enregistrés dans '{results_filename}'\n")


if __name__ == "__main__":

    #Options TEST
    list_mu = np.arange(0, 0.7, 0.1)  # mu between 0 and 0.6

    random.seed(42)  # Fixer la seed
    main(list_mu)

