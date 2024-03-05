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

        # initialize MST,number of vertices, and tracking array

        num_vertices = self.adj_mat.shape[0]
        self.mst = np.zeros_like(self.adj_mat)
        in_mst = [False] * num_vertices

        # initialize priority queue for edges (weight, vertex1, vertex2)

        edges = [(0, 0, i) for i in range(1, num_vertices)]
        heapq.heapify(edges)
        num_edges = 0

        # while not all vertices are included in the MST, work through the queue

        while edges and num_edges < num_vertices - 1:
            weight, start_vertex, end_vertex = heapq.heappop(edges)

            # if end_vertex is not already in MST:

            if not in_mst[end_vertex]:

                # add it, and assign weights in mst

                in_mst[end_vertex] = True
                self.mst[start_vertex, end_vertex] = weight
                self.mst[end_vertex, start_vertex] = weight
                num_edges += 1

                # add new edges to the priority queue

                for next_vertex in range(num_vertices):
                    if self.adj_mat[end_vertex, next_vertex] > 0 and not in_mst[next_vertex]:
                        heapq.heappush(edges, (self.adj_mat[end_vertex, next_vertex], end_vertex, next_vertex))

        # handle disconnected graph case, where the number of vertices -1 is greater than the  number of edges

        if num_edges < num_vertices - 1:
            raise Exception("input graph is disconnected")