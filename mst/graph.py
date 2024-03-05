import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """


        # initialize mst, and track both vertices and edges selected for ,st

        num_vertices = self.adj_mat.shape[0]
        self.mst = np.zeros_like(self.adj_mat)
        in_mst = [False] * num_vertices

        # edges selected for mst (weight, from_vertex, to_vertex)

        edges_selected = []

        # initialize all vertices not in MST with infinite edge cost except for the first one

        edge_costs = [(np.inf, 0, i) for i in range(1, num_vertices)]

        # start at vertex 0 with 0 cost

        edge_costs.insert(0, (0, -1, 0))

        # convert edge costs list into a heap

        heapq.heapify(edge_costs)

        while edge_costs and len(edges_selected) < num_vertices - 1:

            # choose the edge with minimum weight

            cost, from_vertex, to_vertex = heapq.heappop(edge_costs)

            # if the vertex is already in mst, continue

            if in_mst[to_vertex]:
                continue

            # add the vertex in mst

            in_mst[to_vertex] = True

            # add the edge to the list of chosen edges (if not starting vertex)

            if from_vertex >= 0:
                edges_selected.append((from_vertex, to_vertex))

            # update the heap with the new edge costs to vertices not in MST
                
            for next_vertex, edge_weight in enumerate(self.adj_mat[to_vertex]):
                if not in_mst[next_vertex] and edge_weight > 0:
                    heapq.heappush(edge_costs, (edge_weight, to_vertex, next_vertex))

        # build the mst from the chosen edges
        for from_vertex, to_vertex in edges_selected:
            weight = self.adj_mat[from_vertex][to_vertex]
            self.mst[from_vertex][to_vertex] = weight
            self.mst[to_vertex][from_vertex] = weight