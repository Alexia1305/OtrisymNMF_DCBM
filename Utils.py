from scipy.sparse import issparse

import pysbm
import OtrisymNMF
import networkx as nx
import numpy as np
def DC_BM(G, r, objective_function, inference_algo, numTrials=1, init_partition=None,init_method="random", verbosity=1,tri=True,init_seed=None):
    """
       Performs Degree-Corrected Block Model (DCBM) inference using multiple trials with different initializations
       and returns the partition with the highest objective function value.

       Parameters:
       -----------
       G : networkx.Graph
           The input graph to be partitioned.
       r : int
           The number of blocks (or communities) to detect.
       objective_function : callable
           A function that defines the objective to maximize during inference.
       inference_algo : callable
           The inference algorithm used to optimize the objective function.
       numTrials : int, optional (default=1)
           The number of trials for stochastic inference, each with a different initialization.
       init_partition : list, optional (default=None)
           An initial partitioning of nodes (if available). If None, a random partition is used in each trial.
       verbosity : int, optional (default=1)
           Level of verbosity for logging (not used in the function).
        tri : boolean, optional (default=True)
            If tri=True, the node partition list is sorted in ascending order (sorted(G.nodes)).
            If tri=False, the original node order in graph G is preserved.
        init_seed : float, optional (default=None)
            Random seed for the initialization for the experiments


       Returns:
       --------
       list
           A list where each entry corresponds to the block assignment of the respective node in sorted order.
           The partition with the best (highest) objective function value is returned.
       """
    obj_max_d = -float('inf')
    best_degree_partition = pysbm.NxPartition(
        graph=G,
        number_of_blocks=r,
    )
    degree_corrected_objective_function = objective_function(is_directed=False)
    for i in range(numTrials):
        if init_partition is not None:
            if tri==True :
                degree_corrected_partition = pysbm.NxPartition(
                    graph=G,
                    number_of_blocks=r,
                    representation={node: init_partition[i] for i, node in enumerate(sorted(G.nodes))})
            else:
                degree_corrected_partition = pysbm.NxPartition(
                    graph=G,
                    number_of_blocks=r,
                    representation={node: init_partition[i] for i, node in enumerate(G.nodes)})
            obj_value_d = degree_corrected_objective_function.calculate(degree_corrected_partition)
            if obj_value_d > obj_max_d:
                best_degree_partition = degree_corrected_partition
                obj_max_d = obj_value_d
        else:
            if init_method=="SVCA":
                if tri:
                    X = nx.adjacency_matrix(G, nodelist=sorted(G.nodes))
                else:
                    X = nx.adjacency_matrix(G)
                if issparse(X):
                    X = X.toarray()
                if init_seed is not None:
                    init_seed += 10 * i
                W = OtrisymNMF.initialize_W(X,r,method="SVCA",init_seed=init_seed)
                v = np.argmax(W, axis=1)
                if tri == True:
                    degree_corrected_partition = pysbm.NxPartition(
                        graph=G,
                        number_of_blocks=r,
                        representation={node: v[i] for i, node in enumerate(sorted(G.nodes))})
                else:
                    degree_corrected_partition = pysbm.NxPartition(
                        graph=G,
                        number_of_blocks=r,
                        representation={node: v[i] for i, node in enumerate(G.nodes)})

                obj_value_d = degree_corrected_objective_function.calculate(degree_corrected_partition)
                if obj_value_d > obj_max_d:
                    best_degree_partition = degree_corrected_partition
                    obj_max_d = obj_value_d

            else :
                degree_corrected_partition = pysbm.NxPartition(
                    graph=G,
                    number_of_blocks=r,
                )


        degree_corrected_inference = inference_algo(G, degree_corrected_objective_function,
                                                    degree_corrected_partition)
        degree_corrected_inference.infer_stochastic_block_model()
        obj_value_d = degree_corrected_objective_function.calculate(degree_corrected_partition)

        if obj_value_d > obj_max_d:
            best_degree_partition = degree_corrected_partition
            obj_max_d = obj_value_d
    if verbosity:
        print(f"Best logP : {obj_max_d}")
    if tri==True:
        return [best_degree_partition.get_block_of_node(node) for node in sorted(G.nodes)]
    else :
        return [best_degree_partition.get_block_of_node(node) for node in G.nodes]
