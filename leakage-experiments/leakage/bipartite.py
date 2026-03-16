from __future__ import annotations

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import sparse


def build_trade_counterparty_incidence(
    trades: pd.DataFrame,
    counterparty_col: str = "counterparty_id",
    weight_col: Optional[str] = None,
) -> Tuple[sparse.csr_matrix, Dict[int, int], Dict[int, int]]:
    if counterparty_col not in trades.columns:
        raise ValueError(f"Missing column '{counterparty_col}' in trades")

    counterparty_ids = trades[counterparty_col].astype(int).to_numpy()
    uniq_counterparties = np.unique(counterparty_ids)

    n_trades = len(trades)
    n_counterparties = len(uniq_counterparties)

    counterparty_id_to_row = {cid: j for j, cid in enumerate(uniq_counterparties.tolist())}
    trade_index_to_col = {i: i for i in range(n_trades)}

    rows = np.array([counterparty_id_to_row[cid] for cid in counterparty_ids], dtype=int)
    cols = np.arange(n_trades, dtype=int)

    if weight_col is None:
        data = np.ones(n_trades, dtype=float)
    else:
        w = trades[weight_col].to_numpy(dtype=float)
        w = np.where(np.isfinite(w), w, 0.0)
        data = w

    b = sparse.csr_matrix((data, (rows, cols)), shape=(n_counterparties, n_trades))
    return b, counterparty_id_to_row, trade_index_to_col


def build_hybrid_adjacency(
    b: sparse.csr_matrix,
    w_tt: Optional[sparse.csr_matrix] = None,
    alpha_tt: float = 1.0,
    alpha_bt: float = 1.0,
) -> sparse.csr_matrix:
    n_c, n_t = b.shape

    if w_tt is None:
        w_tt = sparse.csr_matrix((n_t, n_t))
    else:
        w_tt = w_tt.tocsr()
        if w_tt.shape != (n_t, n_t):
            raise ValueError(f"w_tt must be shape {(n_t, n_t)} but got {w_tt.shape}")

    zc = sparse.csr_matrix((n_c, n_c))

    top = sparse.hstack([zc, alpha_bt * b], format="csr")
    bot = sparse.hstack([alpha_bt * b.T, alpha_tt * w_tt], format="csr")
    a = sparse.vstack([top, bot], format="csr")

    a = a.tolil()
    a.setdiag(0.0)
    a = a.tocsr()

    return a


def normalized_laplacian_from_adjacency(a: sparse.csr_matrix) -> sparse.csr_matrix:
    a = (a + a.T) * 0.5
    a = a.tocsr().astype(np.float64)

    n = a.shape[0]
    d = np.asarray(a.sum(axis=1)).reshape(-1)
    d_inv_sqrt = np.zeros_like(d)
    mask = d > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])

    d_inv_sqrt_mat = sparse.diags(d_inv_sqrt, format="csr")
    i_mat = sparse.eye(n, format="csr", dtype=np.float64)

    l = i_mat - (d_inv_sqrt_mat @ a @ d_inv_sqrt_mat)
    l = (l + l.T) * 0.5
    return l