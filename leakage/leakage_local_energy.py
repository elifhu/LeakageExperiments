from __future__ import annotations

import numpy as np
from scipy import sparse


def local_energy_from_adjacency(a: sparse.csr_matrix, f: np.ndarray):
    if not sparse.isspmatrix_csr(a):
        a = a.tocsr()

    f = np.asarray(f, dtype=float).reshape(-1)
    n = a.shape[0]

    if f.shape[0] != n:
        raise ValueError("f length must match adjacency size")

    deg = np.asarray(a.sum(axis=1)).reshape(-1)
    af = a @ f
    af2 = a @ (f ** 2)

    e = deg * (f ** 2) - 2.0 * f * af + af2
    e = np.maximum(e, 0.0)

    e_norm = e / (deg + 1e-12)
    return e, e_norm, deg