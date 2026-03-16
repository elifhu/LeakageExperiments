from __future__ import annotations

from typing import Literal, Dict, Optional
import numpy as np
import pandas as pd


Agg = Literal["mean", "sum", "median", "max"]


def _agg(x: np.ndarray, how: Agg) -> float:
    if x.size == 0:
        return float("nan")
    if how == "mean":
        return float(np.mean(x))
    if how == "sum":
        return float(np.sum(x))
    if how == "median":
        return float(np.median(x))
    if how == "max":
        return float(np.max(x))
    raise ValueError(f"Unknown agg: {how}")


def counterparty_scores_lowpass(f_hat: np.ndarray, n_counterparties: int) -> np.ndarray:
    f_hat = np.asarray(f_hat, dtype=float).reshape(-1)
    return f_hat[:n_counterparties].copy()


def counterparty_scores_spiky_residual(
    trades: pd.DataFrame,
    f_true: np.ndarray,
    f_hat_trade: np.ndarray,
    counterparty_col: str = "counterparty_id",
    how: Agg = "mean",
    use_abs: bool = True,
) -> Dict[int, float]:
    f_true = np.asarray(f_true, dtype=float).reshape(-1)
    f_hat_trade = np.asarray(f_hat_trade, dtype=float).reshape(-1)

    if len(trades) != f_true.shape[0] or f_hat_trade.shape[0] != f_true.shape[0]:
        raise ValueError("Length mismatch between trades, f_true, and f_hat_trade")

    resid = f_true - f_hat_trade
    if use_abs:
        resid = np.abs(resid)

    counterparty_ids = trades[counterparty_col].astype(int).to_numpy()
    scores: Dict[int, float] = {}

    for cid in np.unique(counterparty_ids):
        idx = np.where(counterparty_ids == cid)[0]
        scores[int(cid)] = _agg(resid[idx], how)

    return scores


def counterparty_scores_from_trade_values(
    trades: pd.DataFrame,
    trade_values: np.ndarray,
    counterparty_col: str = "counterparty_id",
    how: Agg = "mean",
    use_abs: bool = True,
) -> Dict[int, float]:
    v = np.asarray(trade_values, dtype=float).reshape(-1)
    if len(trades) != v.shape[0]:
        raise ValueError("trade_values length must match trades length")

    if use_abs:
        v = np.abs(v)

    counterparty_ids = trades[counterparty_col].astype(int).to_numpy()
    out: Dict[int, float] = {}

    for cid in np.unique(counterparty_ids):
        idx = np.where(counterparty_ids == cid)[0]
        out[int(cid)] = _agg(v[idx], how)

    return out


def build_stacked_counterparty_trade_signal(
    trades: pd.DataFrame,
    trade_signal: np.ndarray,
    counterparty_id_to_row: Dict[int, int],
    counterparty_col: str = "counterparty_id",
    how: str = "mean",
    use_abs_counterparty: bool = True,
) -> np.ndarray:
    trade_signal = np.asarray(trade_signal, dtype=float).reshape(-1)
    n_t = len(trades)

    if trade_signal.shape[0] != n_t:
        raise ValueError("trade_signal length must match number of trades")

    n_c = len(counterparty_id_to_row)
    f = np.zeros(n_c + n_t, dtype=float)

    v = np.abs(trade_signal) if use_abs_counterparty else trade_signal
    counterparty_ids = trades[counterparty_col].astype(int).to_numpy()

    for cid in np.unique(counterparty_ids):
        idx = np.where(counterparty_ids == cid)[0]
        x = v[idx]

        if how == "mean":
            val = float(np.mean(x))
        elif how == "sum":
            val = float(np.sum(x))
        elif how == "median":
            val = float(np.median(x))
        elif how == "max":
            val = float(np.max(x))
        else:
            raise ValueError(f"Unknown agg: {how}")

        f[counterparty_id_to_row[int(cid)]] = val

    f[n_c:] = trade_signal
    return f


def align_counterparty_metric(
    metric_obj,
    n_counterparties: int,
    counterparty_id_to_row: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    out = np.full(n_counterparties, np.nan, dtype=float)

    if isinstance(metric_obj, dict):
        iterator = metric_obj.items()
    elif isinstance(metric_obj, pd.Series):
        iterator = metric_obj.items()
    else:
        arr = np.asarray(metric_obj).reshape(-1)
        m = min(len(arr), n_counterparties)
        out[:m] = arr[:m]
        return out

    for k, v in iterator:
        if counterparty_id_to_row is None:
            row = int(k)
        else:
            if int(k) not in counterparty_id_to_row:
                continue
            row = counterparty_id_to_row[int(k)]

        if 0 <= row < n_counterparties:
            out[row] = float(v)

    return out


def build_counterparty_vector_table(
    theta_lp,
    scores_hp,
    scores_energy,
    counterparty_id_to_row: Dict[int, int],
    t_diff: float,
) -> pd.DataFrame:
    n_counterparties = len(counterparty_id_to_row)

    s_lp = align_counterparty_metric(theta_lp, n_counterparties, counterparty_id_to_row=None)
    r_hp = align_counterparty_metric(scores_hp, n_counterparties, counterparty_id_to_row=counterparty_id_to_row)
    p_energy = align_counterparty_metric(scores_energy, n_counterparties, counterparty_id_to_row=counterparty_id_to_row)

    row_to_counterparty_id = {row: cid for cid, row in counterparty_id_to_row.items()}

    df = pd.DataFrame({
        "row_id": np.arange(n_counterparties, dtype=int),
        "counterparty_id": [row_to_counterparty_id[i] for i in range(n_counterparties)],
        "S_lp": s_lp,
        "R_hp": r_hp,
        "P_energy": p_energy,
    })

    df["vector_norm"] = np.sqrt(
        np.nan_to_num(df["S_lp"], nan=0.0) ** 2 +
        np.nan_to_num(df["R_hp"], nan=0.0) ** 2 +
        np.nan_to_num(df["P_energy"], nan=0.0) ** 2
    )

    df["t_diff"] = t_diff
    df = df.sort_values("vector_norm", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return df