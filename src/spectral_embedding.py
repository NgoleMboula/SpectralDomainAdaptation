import torch
from typing import List
import warnings
import numpy as np
from typing import Optional, Literal, Union
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from packaging.version import parse as parse_version
#from sklearn.manifold import spectral_embedding as __spectral_embedding
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from src.spectral_utils import (check_array,
                         check_symmetric,
                         check_random_state,
                         _graph_is_connected,
                         _graph_connected_component,
                         _deterministic_vector_sign_flip,
                         _init_arpack_v0,
                         _set_diag,
                         )


def _spectral_embedding(
        adjacency,
        *,
        n_components=8,
        eigen_solver=None,
        random_state=None,
        eigen_tol="auto",
        norm_laplacian=True,
        drop_first=True
):
    """Compute the spectral embedding of a graph represented by its adjacency matrix."""
    
    adjacency = check_symmetric(adjacency)

    if eigen_solver is None:
        eigen_solver = "arpack"
    n_nodes = adjacency.shape[0]
    if drop_first:
        n_components += 1
    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected. Spectral embedding may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian, return_diag=True)    
    eigenvals = None
    if eigen_solver == "arpack" or (
        eigen_solver != "lobpcg"
        and (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)
    ):
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        try:
            tol = 0 if eigen_tol == "auto" else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            laplacian = check_array(laplacian, accept_sparse="csr", accept_large_sparse=False)
            eigenvals, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = "lobpcg"
            laplacian *= -1

    if eigen_solver == "lobpcg":
        laplacian = check_array(laplacian, dtype=[np.float64, np.float32], accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            if sparse.issparse(laplacian):
                laplacian = laplacian.toarray()
            eigenvals, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
            X[:, 0] = dd.ravel()
            tol = None if eigen_tol == "auto" else eigen_tol
            eigenvals, diffusion_map = lobpcg(
                laplacian, X, tol=tol, largest=False, maxiter=2000
            )
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd

    embedding = _deterministic_vector_sign_flip(embedding)
    return embedding[1:n_components].T if drop_first else embedding[:n_components].T, eigenvals



def SPEMB(
    adjacency: Union[np.ndarray, torch.Tensor],
    n_components: int = 2,
    eigen_solver: Optional[str] = None,
    random_state: Optional[int] = None,
    eigen_tol: float = 1e-10,
    norm_laplacian: bool = True,
    drop_first: bool = True) -> torch.Tensor:
    """
    Wrapper function for Computing the spectral embedding of a graph adjacency matrix.
    """

    if isinstance(adjacency, torch.Tensor):
        adjacency_np = adjacency.detach().cpu().numpy()
    elif isinstance(adjacency, np.ndarray):
        adjacency_np = adjacency
    else:
        raise TypeError("`adjacency` must be a torch.Tensor or numpy.ndarray.")

    if adjacency_np.ndim != 2 or adjacency_np.shape[0] != adjacency_np.shape[1]:
        raise ValueError(f"Input adjacency must be a square matrix, got shape {adjacency_np.shape}.")
    
    random_state = check_random_state(random_state)
    embedding, eigenvals = _spectral_embedding(
        adjacency_np,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=norm_laplacian,
        drop_first=drop_first
    )

    return torch.from_numpy(embedding.copy()).float(), eigenvals


def adjacency_matrix___(G_bar):
    "Constructs a block adjacency matrix from the transport plans between sources, barycenter, and target."

    source_to_barycenter = G_bar['source_to_barycenter']
    barycenter_to_target = G_bar['barycenter_to_target']
    n_sources = len(source_to_barycenter)
    couplings = []
    sizes_d = []
    sizes_b = []

    for i in range(n_sources):
        coupling = source_to_barycenter[f'Barycenter Coupling {i}']
        couplings.append(coupling.T)  
        sizes_d.append(coupling.shape[0]) 
        sizes_b.append(coupling.shape[1]) 

    couplings.append(barycenter_to_target)
    sizes_d.append(barycenter_to_target.shape[1])
    sizes_b.append(barycenter_to_target.shape[0])

    b = max(sizes_b)
    total_size = b + sum(sizes_d)
    A = np.zeros((total_size, total_size))

    current_col = b
    for idx, coupling in enumerate(couplings):
        d_i = coupling.shape[1]
        A[:b, current_col:current_col+d_i] = coupling  
        current_col += d_i

    current_row = b
    for idx, coupling in enumerate(couplings):
        d_i = coupling.shape[1]
        A[current_row:current_row+d_i, :b] = coupling.T
        current_row += d_i

    return A

def adjacency_bipartite(G_bar):
    n_sources = len(G_bar) - 1  
    d, b = G_bar[0].shape
    for plan in G_bar:
        assert plan.shape == (d, b), "All OT plans must have shape (d, b)"

    total_size = b + (n_sources + 1) * d
    A = np.zeros((total_size, total_size))
    current_row = b
    for plan in G_bar:
        A[current_row:current_row + d, :b] = plan  
        current_row += d

    current_col = b
    for plan in G_bar:
        A[:b, current_col:current_col + d] = plan.T
        current_col += d

    return A

