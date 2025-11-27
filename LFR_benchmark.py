from matplotlib import pyplot as plt
from scipy.sparse import diags

import OtrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
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
            "OtrisymNMF": {"NMI": [], "AMI": [], "Time": []},
            "KN": {"NMI": [], "AMI": [], "Time": []},
            "KL_EM": {"NMI": [], "AMI": [], "Time": []},
            "MHA250k": {"NMI": [], "AMI": [], "Time": []},
            "OtrisymNMF_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "SVCA": {"NMI": [], "AMI": [], "Time": []},
            "KN_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "KL_EM_SVCA": {"NMI": [], "AMI": [], "Time": []},
            "MHA250k_SVCA": {"NMI": [], "AMI": [], "Time": []},

        }

        for idx, G in enumerate(graphs, start=1):


            print(f"Processed {idx} out of {len(graphs)} graphs.")

            labels = [G.nodes[v]['community'] for v in G.nodes]
            r = max(labels)

            # OtrisymNMF
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best,_ = OtrisymNMF.OtrisymNMF_CD(X, r, numTrials=10, init_method="random",
                                                                          verbosity=0, init_seed=idx, delta=1e-5,)
            end_time = time.time()
            results["OtrisymNMF"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["OtrisymNMF"]["AMI"].append(adjusted_mutual_info_score(labels,v_best,average_method='max'))
            results["OtrisymNMF"]["Time"].append(end_time - start_time)

            # # KN
            # start_time = time.time()
            # KN_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
            #                         numTrials=10,
            #                         init_method="random", verbosity=0)
            # end_time = time.time()
            # results["KN"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            # results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            # results["KN"]["Time"].append(end_time - start_time)
            #
            #
            # # KL_EM
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="random", verbosity=0)
            # end_time = time.time()
            # results["KL_EM"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            # results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            # results["KL_EM"]["Time"].append(end_time - start_time)
            #
            #
            # # MHA250
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,init_method="random",
            #                       verbosity=0)
            # end_time = time.time()
            #
            # results["MHA250k"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            # results["MHA250k"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            # results["MHA250k"]["Time"].append(end_time - start_time)


            #OtrisymNMF_SVCA
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            w_best, v_best, S_best, error_best,_ = OtrisymNMF.OtrisymNMF_CD(X, r,numTrials=10,init_method="SVCA",verbosity=0, init_seed=idx,delta=1e-5)
            end_time = time.time()

            results["OtrisymNMF_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["OtrisymNMF_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best,average_method='max'))
            results["OtrisymNMF_SVCA"]["Time"].append(end_time - start_time)


            #SVCA only
            start_time = time.time()
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            w_best, v, S_best, error_best = OtrisymNMF.Community_detection_SVCA(X, r, numTrials=10, verbosity=0)
            end_time = time.time()
            results["SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v))
            results["SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v,average_method='max'))
            results["SVCA"]["Time"].append(end_time - start_time)

            # # KL_EM initialized by SVCA
            # start_time = time.time()
            # EM_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
            #                      init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # results["KL_EM_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            # results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            # results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            #
            # # KN initialized by SVCA
            # start_time = time.time()
            # KN_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=10,
            #                      init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # results["KN_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            # results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            # results["KN_SVCA"]["Time"].append(end_time - start_time)
            #
            #
            # # MHA250 initialized by SVCA
            # start_time = time.time()
            # MHA_partition = DC_BM(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
            #                       pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,
            #                       init_method="SVCA", verbosity=0, init_seed=idx)
            # end_time = time.time()
            # results["MHA250k_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            # results["MHA250k_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            # results["MHA250k_SVCA"]["Time"].append(end_time - start_time)

        summary = {}
        for algo, data in results.items():
            summary[algo] = {
                "NMI moyen": np.round(np.mean(data["NMI"]), 4),
                "Erreur type NMI": np.round(np.std(data["NMI"], ddof=1), 4),
                "AMI moyen": np.round(np.mean(data["AMI"]), 4),
                "Erreur type AMI": np.round(np.std(data["AMI"], ddof=1), 4),
                "Temps moyen (s)": np.round(np.mean(data["Time"]), 2),
                "Erreur type Temps": np.round(np.std(data["Time"], ddof=1), 2)
            }
        df_results = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\nRésultats pour mu={mu:.1f}:")
        # Results Display
        print(df_results)

        # Sauvegarde des résultats dans un fichier CSV
        results_filename = f"mu_{mu:.1f}_resultsAMIverif.csv"
        df_results.to_csv(results_filename)
        print(f"Résultats enregistrés dans '{results_filename}'\n")

def displayLFR(mu):
    """ Test LFR benchmark """
    n = 1000

    graphs_folder = f"Data/LFR/mu_{mu:.1f}"
    graphs = read_graphs_from_files(graphs_folder, n)
    G=graphs[1].copy()
    labels = [G.nodes[v]['community'] for v in G.nodes]
    r = max(labels)
    # Palette de 20 couleurs distinctes
    cmap = plt.cm.get_cmap("tab20", r)
    colors = [cmap(i) for i in range(r)]
    for u, v in G.edges():
        if G.nodes[u]['community'] == G.nodes[v]['community']:
            G[u][v]['weight'] = 1.2  # poids fort (attire +)
            G[u][v]['intra'] = True
        else:
            G[u][v]['weight'] = 0.2  # poids faible (repousse)
            G[u][v]['intra'] = False

    # Formes disponibles (on les réutilisera si r > len(shapes))
    shapes = ['o', 's', '^', 'v', 'D', 'p', 'h', '8', '*']

    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.15)  # layout du graphe

    plt.figure(figsize=(14, 14))

    for i in range(r):
        nodes_i = [node for node in G.nodes() if G.nodes[node]['community'] == i+1]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_i,
            node_color=[colors[i]],
            node_shape=shapes[i % len(shapes)],  # recycle les formes
            label=f'Comm {i}',
            node_size=40
        )

    intra_edges = [(u, v) for u, v in G.edges() if G[u][v]['intra']]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=intra_edges,
        width=0.3,
        alpha=0.3,
        style="solid"
    )

    # --- Arêtes inter (plus fines, en pointillés) ---
    inter_edges = [(u, v) for u, v in G.edges() if not G[u][v]['intra']]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=inter_edges,
        width=0,
        alpha=0,
        style="solid"
    )

    plt.axis('off')
    # Sauvegarde en PNG haute résolution
    plt.savefig(f"LFR_mu_{mu:.1f}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #displayLFR(0.1)
    #Options TEST
    list_mu = np.arange(0.2, 0.3, 0.1)  # mu between 0 and 0.6

    random.seed(42)  # Fixer la seed
    main(list_mu)

