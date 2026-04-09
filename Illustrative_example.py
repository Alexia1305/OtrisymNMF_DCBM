import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
W=np.array([[1,0,0],[2,0,0],[1,0,0],[0,1,0],[0,2,0],[0,1,0],[0,0,1],[0,0,1],[0,0,2],[0,1,1]])
S=np.array([[5,1,1],[1,5,1],[1,1,5]])
X=W@S@W.T
print(X)
G = nx.from_numpy_array(X)
mapping = {i: i+1 for i in G.nodes()}
G = nx.relabel_nodes(G, mapping)
G.remove_edges_from(nx.selfloop_edges(G))
pos = nx.spring_layout(G,seed=12)
# Extraire les poids pour les utiliser comme largeur des arêtes
weights = [G[u][v]['weight'] for u, v in G.edges()]
# Mettre à l'échelle les poids pour largeur des arêtes (par exemple entre 0.5 et 5)
min_width = 0.2
max_width = 5
min_weight = min(weights)
max_weight = max(weights)

# Fonction de mise à l'échelle linéaire
scaled_weights = [
    min_width + (w - min_weight) / (max_weight - min_weight) * (max_width - min_width)
    if max_weight != min_weight else 1
    for w in weights
]
# Tracer les noeuds
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)

# Tracer les arêtes
# Tracer les arêtes avec largeur proportionnelle au poids
nx.draw_networkx_edges(G, pos, width=scaled_weights)

# Ajouter les labels des noeuds
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')



# Afficher
plt.axis('off')  # enlève les axes
plt.savefig("graph.png", dpi=300, bbox_inches='tight')
plt.show()
