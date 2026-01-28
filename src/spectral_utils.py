import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from packaging.version import parse as parse_version
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
sp_version = parse_version(np.__version__)

def check_array(array, accept_sparse=False, dtype=None, accept_large_sparse=True):
    "Check input array, optionally converting it to a dense NumPy array."

    if sparse.issparse(array):
        if not accept_sparse:
           raise TypeError("A sparse matrix was passed, but dense data required.")
        return array
    return np.array(array, dtype=dtype, copy=False)

def check_random_state(seed):

    "Generates a reproducible NumPy RandomState object from an integer seed,"
    " None, or an existing RandomState instance."

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)

def check_symmetric(array, tol=1e-10):
    "Verifies whether a dense or sparse matrix is symmetric within a given numerical tolerance."

    if sparse.issparse(array):
        diff = ( array - array.T).tocoo()
        if diff.nnz == 0 or np.all(np.abs(diff.data)<tol):
            return array
        raise ValueError("Matrix is not symmetric.")
    else:
        if np.allclose(array, array.T, atol=tol):
            return array
        raise ValueError("Matrix is not symmetric.")
    
def _graph_connected_component(graph, node_id):
    "Computes the set of nodes reachable from a specified node in a graph by iteratively exploring neighbors."

    n_node = graph.shape[0]
    if sparse.issparse(graph):
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            neighbors = graph[[i], :].toarray().ravel() if sparse.issparse(graph) else graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes 

def _graph_is_connected(graph):
    "Checks if all nodes in a graph belong to a single connected component, supporting both sparse and dense representations."
    if sparse.issparse(graph):
        accept_large_sparse = sp_version >= parse_version("1.11.3")
        graph = check_array(graph, acceptsparse=True, accept_large_sparse=accept_large_sparse)
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]
    
def _set_diag(laplacian, value, norm_laplacian):
    "Sets the diagonal elements of a Laplacian matrix to a specified value, with special handling for normalized Laplacians."
    n_nodes = laplacian.shape[0]
    if not sparse.issparse(laplacian):
        if norm_laplacian:
            laplacian.flat[::n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] == value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        laplacian = laplacian.todia() if n_diags <= 7 else laplacian.tocsr()
    return laplacian

def _init_arpack_v0(n, random_state):
    "Generates a random initial vector for ARPACK eigenvalue computations, ensuring reproducibility through a given random state."
    v0 = random_state.uniform(-1, 1, n)
    return v0

def _deterministic_vector_sign_flip(u):
    "Ensures deterministic sign orientation of eigenvectors by flipping them based on the sign of their largest absolute component."
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    signs[signs==0] = 1
    u *= signs
    return u


