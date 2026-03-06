import pysbm
import networkx as nx
import matplotlib.pylab as pl
from dcbm import dcbm
import otrisymNMF
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
import random
# Reading and displaying the graph

graph = nx.karate_club_graph()
club_labels = {node: 1 if graph.nodes[node]['club'] == 'Mr. Hi' else 0 for node in graph.nodes}
club_vector = np.array([club_labels[node] for node in graph.nodes])
position = nx.spring_layout(graph,seed=23)
node_sizes = [10+graph.degree(node) * 40 for node in graph]  # Facteur d'échelle pour bien voir les tailles


#%%
X = nx.adjacency_matrix(graph, nodelist=graph.nodes)
np.random.seed(100)

w, v, S, error,_ = otrisymNMF.frost(X, 2,numTrials=10,verbosity=1,init_seed=2)

print("NMI score:",nmi(club_vector,v))

random.seed(50)
np.random.seed(12)
dcbm_partition=dcbm(graph,2,pysbm.DegreeCorrectedUnnormalizedLogLikelyhood,pysbm.KerninghanLinInference, numTrials=10)
print("NMI score:",nmi(club_vector,dcbm_partition))
print("The node misclassified additionally compared to OtrisymNMF:",[i for i,value in enumerate(v != dcbm_partition) if value])
