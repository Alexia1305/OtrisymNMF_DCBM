import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import OtrisymNMF
graph = nx.karate_club_graph()
club_labels = {node: 1 if graph.nodes[node]['club'] == 'Mr. Hi' else 0 for node in graph.nodes}
X = nx.adjacency_matrix(graph, nodelist=graph.nodes)
np.random.seed(100)
w, v, S, error = OtrisymNMF.OtrisymNMF_CD(X, 2,numTrials=100,verbosity=0,init_seed=2)
w2, v2, S2, error2 = OtrisymNMF.OtrisymNMF_CD(X.toarray(), 2,numTrials=100,verbosity=0,init_seed=2)
print('ok')
import pysbm

pysbm.EMInference(G,)
