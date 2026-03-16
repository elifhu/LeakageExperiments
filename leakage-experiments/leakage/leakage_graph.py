from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import eigsh, ArpackNoConvergence


def build_kernels(trades, tau: float, gamma: float):
    x_state = trades[["sigma", "volume_proxy", "liq"]].values.astype(float)
    xz = StandardScaler().fit_transform(x_state)

    t0 = trades["time"].iloc[0]
    tt = (trades["time"] - t0).dt.total_seconds().values.astype(float)

    dt_mat = tt.reshape(-1, 1) - tt.reshape(1, -1)
    k_time = np.exp(-(dt_mat ** 2) / (tau ** 2))

    diff_state = xz[:, None, :] - xz[None, :, :]
    dist2_state = np.sum(diff_state ** 2, axis=2)
    k_state = np.exp(-dist2_state / (gamma ** 2))

    return k_time, k_state, dist2_state, tt, xz


def knn_mask(dist2_state: np.ndarray, k: int, symmetric: bool = True) -> np.ndarray:
    n = dist2_state.shape[0]
    idx = np.argpartition(dist2_state, kth=k + 1, axis=1)[:, :k + 1]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, idx] = True
    np.fill_diagonal(mask, False)
    return (mask | mask.T) if symmetric else mask


def build_graph_w(k_time: np.ndarray, k_state: np.ndarray, mask: np.ndarray) -> sp.csr_matrix:
    w = (k_time * k_state) * mask
    np.fill_diagonal(w, 0.0)
    return sp.csr_matrix(w)


def normalized_laplacian(w: sp.csr_matrix) -> sp.csr_matrix:
    d = np.array(w.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-12), 0, format="csr")
    i_mat = sp.eye(w.shape[0], format="csr")
    l = i_mat - d_inv_sqrt @ w @ d_inv_sqrt
    return l


def dirichlet_energy(laplacian: sp.csr_matrix, f: np.ndarray) -> float:
    f = f.astype(float).reshape(-1)
    return float(f @ (laplacian @ f))


def local_energy(w: sp.csr_matrix, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = f.astype(float).reshape(-1)
    wf = w @ f
    wf2 = w @ (f ** 2)
    d = np.array(w.sum(axis=1)).ravel()

    e_local = d * (f ** 2) - 2.0 * f * wf + wf2
    e_local_norm = e_local / (d + 1e-12)

    return e_local, e_local_norm, d


def strong_edge_subgraph(w: sp.csr_matrix, quantile_thr: float = 0.85) -> sp.csr_matrix:
    wcoo = w.tocoo()
    if wcoo.nnz == 0:
        return w.copy()

    thr = np.quantile(wcoo.data, quantile_thr)
    keep = wcoo.data >= thr
    w_strong = sp.csr_matrix(
        (wcoo.data[keep], (wcoo.row[keep], wcoo.col[keep])),
        shape=w.shape,
    )
    return w_strong


def largest_connected_component(w: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    n_comp, labels = csgraph.connected_components(w, directed=False)
    if n_comp <= 1:
        keep_nodes = np.arange(w.shape[0])
        return w.tocsr(), keep_nodes

    sizes = np.bincount(labels)
    giant = int(np.argmax(sizes))
    keep_nodes = np.where(labels == giant)[0]
    w_g = w[keep_nodes][:, keep_nodes].tocsr()
    return w_g, keep_nodes


def eigmodes(laplacian: sp.csr_matrix, k: int = 6):
    n = laplacian.shape[0]
    k = min(k, max(1, n - 2))

    try:
        evals, evecs = eigsh(laplacian, k=k, which="SM", tol=1e-4, maxiter=20000)
        idx = np.argsort(evals)
        return evals[idx], evecs[:, idx]

    except ArpackNoConvergence:
        try:
            evals, evecs = eigsh(laplacian, k=k, sigma=0.0, which="LM", tol=1e-4, maxiter=20000)
            idx = np.argsort(evals)
            return evals[idx], evecs[:, idx]

        except Exception:
            l_dense = laplacian.toarray()
            evals, evecs = np.linalg.eigh(l_dense)
            idx = np.argsort(evals)[:k]
            return evals[idx], evecs[:, idx]