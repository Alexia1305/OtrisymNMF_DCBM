import matplotlib
import matplotlib.pyplot as plt
import otrisymNMF
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import pysbm
import time
import numpy as np
import pandas as pd
from dcbm_inference import dcbm,dcbm_PAH
import random
import clustering_mi as cmi # for the assymetrically normalized of the reduced mutual information
import json

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
            "FROST": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "KN": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "KL_EM": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "MHA250k": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "FROST_SVCA": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "SVCA": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "KN_SVCA": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "KL_EM_SVCA": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "MHA250k_SVCA": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []},
            "PAH": {"NMI": [], "AMI": [], "ARMI": [], "Time": [], "CPU Time": []}

        }

        for idx, G in enumerate(graphs, start=1):


            print(f"Processed {idx} out of {len(graphs)} graphs.")

            labels = [G.nodes[v]['community'] for v in G.nodes]
            r = max(labels)

            # FROST
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            start_CPU = time.process_time()
            w_best, v_best, S_best, error_best,_ = otrisymNMF.frost(X, r, numTrials=10, init_method="random",
                                                                            verbosity=0, init_seed=idx, delta=1e-5, )
            end_CPU = time.process_time()
            end_time = time.time()
            results["FROST"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["FROST"]["AMI"].append(adjusted_mutual_info_score(labels, v_best,average_method='max'))
            results["FROST"]["ARMI"].append(cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
            results["FROST"]["Time"].append(end_time - start_time)
            results["FROST"]["CPU Time"].append(end_CPU - start_CPU)

            # KN
            start_time = time.time()
            start_CPU = time.process_time()
            KN_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference,
                                    numTrials=10,init_seed=idx,
                                    init_method="random", verbosity=0)
            end_CPU = time.process_time()
            end_time = time.time()
            results["KN"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            results["KN"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            results["KN"]["ARMI"].append(cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
            results["KN"]["Time"].append(end_time - start_time)
            results["KN"]["CPU Time"].append(end_CPU - start_CPU)


            # KL_EM
            start_time = time.time()
            start_CPU = time.process_time()
            EM_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
                                 init_method="random",init_seed=idx, verbosity=0)
            end_CPU = time.process_time()
            end_time = time.time()
            results["KL_EM"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            results["KL_EM"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            results["KL_EM"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
            results["KL_EM"]["Time"].append(end_time - start_time)
            results["KL_EM"]["CPU Time"].append(end_CPU - start_CPU)


            # MHA250
            start_time = time.time()
            start_CPU = time.process_time()
            MHA_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
                                  pysbm.MetropolisHastingInferenceTwoHundredFiftyK,init_seed=idx, numTrials=10,init_method="random",
                                  verbosity=0)
            end_CPU = time.process_time()
            end_time = time.time()

            results["MHA250k"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            results["MHA250k"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            results["MHA250k"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, MHA_partition, variation="reduced", normalization="first"))
            results["MHA250k"]["Time"].append(end_time - start_time)
            results["MHA250k"]["CPU Time"].append(end_CPU - start_CPU)


            #FROST_SVCA
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            start_time = time.time()
            start_CPU = time.process_time()
            w_best, v_best, S_best, error_best,_ = otrisymNMF.frost(X, r, numTrials=10, init_method="SVCA", verbosity=0, init_seed=idx, delta=1e-5)
            end_CPU = time.process_time()
            end_time = time.time()

            results["FROST_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v_best))
            results["FROST_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v_best,average_method='max'))
            results["FROST_SVCA"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, v_best, variation="reduced", normalization="first"))
            results["FROST_SVCA"]["Time"].append(end_time - start_time)
            results["FROST_SVCA"]["CPU Time"].append(end_CPU - start_CPU)


            #SVCA only
            start_time = time.time()
            start_CPU = time.process_time()
            X = nx.adjacency_matrix(G, nodelist=G.nodes)
            w_best, v, S_best, error_best = otrisymNMF.community_detection_svca(X, r, numTrials=10, verbosity=0,init_seed=idx)
            end_CPU = time.process_time()
            end_time = time.time()
            results["SVCA"]["NMI"].append(normalized_mutual_info_score(labels, v))
            results["SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, v,average_method='max'))
            results["SVCA"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, v, variation="reduced", normalization="first"))
            results["SVCA"]["Time"].append(end_time - start_time)
            results["SVCA"]["CPU Time"].append(end_CPU - start_CPU)

            # KL_EM initialized by SVCA
            start_time = time.time()
            start_CPU = time.process_time()
            EM_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.EMInference, numTrials=10,
                                 init_method="SVCA", verbosity=0, init_seed=idx)
            end_CPU = time.process_time()
            end_time = time.time()
            results["KL_EM_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, EM_partition))
            results["KL_EM_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, EM_partition,average_method='max'))
            results["KL_EM_SVCA"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, EM_partition, variation="reduced", normalization="first"))
            results["KL_EM_SVCA"]["Time"].append(end_time - start_time)
            results["KL_EM_SVCA"]["CPU Time"].append(end_CPU - start_CPU)

            # KN initialized by SVCA
            start_time = time.time()
            start_CPU = time.process_time()
            KN_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, pysbm.KarrerInference, numTrials=10,
                                 init_method="SVCA", verbosity=0, init_seed=idx)
            end_CPU = time.process_time()
            end_time = time.time()
            results["KN_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, KN_partition))
            results["KN_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, KN_partition,average_method='max'))
            results["KN_SVCA"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, KN_partition, variation="reduced", normalization="first"))
            results["KN_SVCA"]["Time"].append(end_time - start_time)
            results["KN_SVCA"]["CPU Time"].append(end_CPU - start_CPU)


            # MHA250 initialized by SVCA
            start_time = time.time()
            start_CPU = time.process_time()
            MHA_partition = dcbm(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,
                                  pysbm.MetropolisHastingInferenceTwoHundredFiftyK, numTrials=10,
                                  init_method="SVCA", verbosity=0, init_seed=idx)
            end_CPU = time.process_time()
            end_time = time.time()
            results["MHA250k_SVCA"]["NMI"].append(normalized_mutual_info_score(labels, MHA_partition))
            results["MHA250k_SVCA"]["AMI"].append(adjusted_mutual_info_score(labels, MHA_partition,average_method='max'))
            results["MHA250k_SVCA"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, MHA_partition, variation="reduced", normalization="first"))
            results["MHA250k_SVCA"]["Time"].append(end_time - start_time)
            results["MHA250k_SVCA"]["CPU Time"].append(end_CPU - start_CPU)

            #PAH
            start_time = time.time()
            start_CPU = time.process_time()
            PAH_partition = dcbm_PAH(G, r, pysbm.DegreeCorrectedUnnormalizedLogLikelyhood, numTrials=10, verbosity=0, init_seed=idx)
            end_CPU = time.process_time()
            end_time = time.time()
            results["PAH"]["NMI"].append(normalized_mutual_info_score(labels, PAH_partition))
            results["PAH"]["AMI"].append(
                adjusted_mutual_info_score(labels, PAH_partition, average_method='max'))
            results["PAH"]["ARMI"].append(
                cmi.normalized_mutual_information(labels, PAH_partition, variation="reduced", normalization="first"))
            results["PAH"]["Time"].append(end_time - start_time)
            results["PAH"]["CPU Time"].append(end_CPU - start_CPU)



        summary = {}
        for algo, data in results.items():
            summary[algo] = {
                "NMI moyen": np.round(np.mean(data["NMI"]), 4),
                "Erreur type NMI": np.round(np.std(data["NMI"], ddof=1), 4),
                "AMI moyen": np.round(np.mean(data["AMI"]), 4),
                "Erreur type AMI": np.round(np.std(data["AMI"], ddof=1), 4),
                "ARMI moyen": np.round(np.mean(data["ARMI"]), 4),
                "Erreur type ARMI": np.round(np.std(data["ARMI"], ddof=1), 4),
                "Temps moyen (s)": np.round(np.mean(data["Time"]), 2),
                "Erreur type Temps": np.round(np.std(data["Time"], ddof=1), 2),
                "Temps moyen CPU (s)": np.round(np.mean(data["CPU Time"]), 2),
                "Erreur type Temps CPU": np.round(np.std(data["CPU Time"], ddof=1), 2)
            }

        with open(f"results/mu_{mu:.1f}_results_f.json", "w") as f:
            json.dump(results, f, indent=4)

        df_results = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\nRésultats pour mu={mu:.1f}:")
        # Results Display
        print(df_results)

        # Sauvegarde des résultats dans un fichier CSV
        results_filename = f"mu_{mu:.1f}_resultstotal.csv"
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
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colors = [cmap(i / max(r-1, 1)) for i in range(r)]
    for u, v in G.edges():
        if G.nodes[u]['community'] == G.nodes[v]['community']:
            G[u][v]['weight'] = 1.2  # poids fort (attire +)
            G[u][v]['intra'] = True
        else:
            G[u][v]['weight'] = 0.2  # poids faible (repousse)
            G[u][v]['intra'] = False

    # Formes disponibles (on les réutilisera si r > len(shapes))
    shapes = ['o', 's', '^', 'v', 'D', 'p', 'h', '8', '*']

    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.17)  # layout du graphe

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
        alpha=0.4,
        style="solid"
    )

    # --- Arêtes inter (plus fines, en pointillés) ---
    inter_edges = [(u, v) for u, v in G.edges() if not G[u][v]['intra']]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=inter_edges,
        width=0.3,
        alpha=0.3,
        style="solid"
    )

    plt.axis('off')
    # Sauvegarde en PNG haute résolution
    plt.savefig(f"LFR_mu_{mu:.1f}.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    #displayLFR(0.1)
    #Options TEST
    list_mu = np.arange(0, 0.5, 0.1)  # mu between 0 and 0.6

    random.seed(42)  # Fixer la seed
    main(list_mu)

