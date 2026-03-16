from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def heat_diffusion_filter(
    laplacian: sparse.csr_matrix,
    f: np.ndarray,
    t: float,
    k_eigs: int = 80,
    which: str = "SM",
) -> Tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f, dtype=float).reshape(-1)
    n = laplacian.shape[0]

    if f.shape[0] != n:
        raise ValueError("f length must match laplacian size")

    k = int(min(k_eigs, n - 2)) if n > 2 else 1
    if k < 1:
        return f.copy(), np.array([0.0])

    evals, evecs = eigsh(laplacian, k=k, which=which)

    g = np.exp(-t * evals)
    alpha = evecs.T @ f
    f_hat = evecs @ (g * alpha)

    return f_hat, evals