import networkx as nx
import networkx as nx

import networkx as nx

import networkx as nx


def convert_networkx_to_dict(G):
    """
    Convertit un graphe NetworkX en un dictionnaire avec les nœuds et arêtes sous un format compatible pour l'écriture en GML.

    Args:
        G (networkx.Graph or networkx.DiGraph): Le graphe NetworkX à convertir.

    Returns:
        dict: Le graphe sous forme de dictionnaire avec les clés 'nodes' et 'edges'.
    """
    nodes_dict = {}
    for node, attributes in G.nodes(data=True):
        # On s'assure que l'attribut 'label' ne soit pas inclus
        node_data = {key: value for key, value in attributes.items() if key != 'label'}
        # Ajouter un attribut 'name' si nécessaire, ici on utilise le 'node' comme nom
        node_data['name'] = node_data.get('name', f"Node_{node}")  # Défaut à "Node_{id}"
        nodes_dict[node] = node_data

    edges_list = list(G.edges())

    return {'nodes': nodes_dict, 'edges': edges_list}


def write_gml_manual_with_name(G, filename):
    """
    Écrit un graphe au format GML sans utiliser NetworkX, en ajoutant l'attribut 'name'
    et en évitant d'ajouter l'attribut 'label'. Les nœuds auront un attribut 'id'.

    Args:
        G (dict): Un dictionnaire représentant le graphe, avec 'nodes' et 'edges'.
        filename (str): Le nom du fichier GML de sortie.
    """
    with open(filename, 'w') as f:
        f.write('graph\n[\n')

        # Écrire les nœuds
        for node, attributes in G['nodes'].items():
            f.write(f'  node\n  [\n    id {node}\n')  # Écrire l'id du nœud

            # Ajouter l'attribut 'name' et autres attributs sans le 'label'
            if 'name' in attributes:
                f.write(f'    name "{attributes["name"]}"\n')
            for key, value in attributes.items():
                if key not in ['label', 'name']:  # Ne pas ajouter 'label' ni répéter 'name'
                    f.write(f'    {key} {value}\n')
            f.write('  ]\n')

        # Écrire les arêtes
        for edge in G['edges']:
            f.write(f'  edge\n  [\n    source {edge[0]}\n    target {edge[1]}\n  ]\n')

        f.write(']\n')

    print(f"Graphe écrit dans '{filename}' avec succès.")


# Lecture du fichier GML
filename = 'Data/polblogs.gml'

# Initialisation des structures de données
edges = []  # Liste pour stocker les arêtes
nodes = {}  # Dictionnaire pour stocker les informations sur les nœuds

# Lire le fichier GML ligne par ligne
with open(filename, 'r') as f:
    lines = f.readlines()

    current_node = None
    current_edge = None
    for line in lines:
        line = line.strip()  # Enlever les espaces superflus

        if line.startswith('node'):  # Détecter un nœud
            current_node = {}  # Initialiser les données pour le nœud
        elif line.startswith('id') and current_node is not None:  # Extraire l'ID du nœud
            current_node['id'] = int(line.split()[1])
        elif line.startswith('label') and current_node is not None:  # Extraire le label du nœud
            current_node['name'] = line.split('"')[1]
        elif line.startswith('value') and current_node is not None:  # Extraire la valeur du nœud
            current_node['value'] = float(line.split()[1])
        elif line == ']' and current_node is not None:  # Fin de la définition du nœud
            nodes[current_node['id']] = current_node  # Ajouter le nœud au dictionnaire

            current_node = None  # Réinitialiser le nœud pour le prochain
        if line.startswith('edge'):  # Détecter un nœud
            current_edge = {}  # Initialiser les données pour le nœud
        elif line.startswith('source') and current_edge is not None:  # Extraire l'ID du nœud
            current_edge['source'] = int(line.split()[1])
        elif line.startswith('target') and current_edge is not None:  # Extraire le label du nœud
            current_edge['target'] = int(line.split()[1])

        elif line == ']' and current_edge is not None:  # Fin de la définition du nœud
            if not [current_edge["source"],current_edge["target"]] in edges and not [current_edge["target"],current_edge["source"]] in edges and current_edge["target"] != current_edge["source"]  :
                edges.append([current_edge["source"],current_edge["target"]]) # Ajouter le nœud au dictionnaire
            current_edge = None  # Réinitialiser le nœud pour le prochain




# Construire le graphe
G = nx.Graph()  # Créer un graphe non dirigé

# Ajouter les nœuds au graphe
for node_id, node_data in nodes.items():
    G.add_node(node_id, name=node_data['name'], value=node_data['value'])

# Ajouter les arêtes au graphe
G.add_edges_from(edges)

# Identifier les nœuds isolés (degré 0)
isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]

# Supprimer les nœuds isolés
G.remove_nodes_from(isolated_nodes)

# Supprimer les nœuds non connectés spécifiques
G.remove_nodes_from([182, 666])

# Vérifier l'état final du graphe
print("Graphe après nettoyage:")
print(nx.info(G))
G_dict = convert_networkx_to_dict(G)
# Sauvegarder le graphe nettoyé en format GML
write_gml_manual_with_name(G_dict, "polblogs_cleaned.gml")

