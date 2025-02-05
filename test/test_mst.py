import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    # check the number of edges, which should be one less than the number of vvertices

    num_vertices = mst.shape[0]
    num_edges = np.count_nonzero(mst) // 2  # Each edge is counted twice
    assert num_edges == num_vertices - 1, 'Proposed MST does not have the correct number of edges'

   # check the  weight, which should be half the sum of mst
                
    total = np.sum(mst) / 2
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # use a dfs and check that the MST construction connectivity

    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()

    num_vertices = g.mst.shape[0]

    def dfs(mst, vertex, visited):
        visited[vertex] = True
        for i, weight in enumerate(mst[vertex]):
            if weight != 0 and not visited[i]:
                dfs(mst, i, visited)

    visited = [False] * num_vertices
    dfs(g.mst, 0, visited)
    assert all(visited), 'Proposed MST is not connected'

    # check the symmetry of both the adj matrix and mst matrix

    def check_symmetry(matrix):
        return np.all(matrix == matrix.T)
    
    assert check_symmetry(g.adj_mat), 'The adjacency matrix of the full graph is not symmetric'
    assert check_symmetry(g.mst), 'The adjacency matrix of the MST is not symmetric'
