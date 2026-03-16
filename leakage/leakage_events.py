from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List


def compute_liq_proxy_from_lookback(df_30s: pd.DataFrame, t: pd.Timestamp, lookback_s: int) -> float:
    lb = df_30s.loc[t - pd.Timedelta(seconds=lookback_s): t - pd.Timedelta(seconds=1)]
    if len(lb) < 10:
        return np.nan

    r = lb["dlog"].dropna()
    if len(r) < 10:
        return np.nan

    vol = float(r.std(ddof=1)) + 1e-12
    mar = float(r.abs().median())
    return float(mar / vol)


def detect_trades(df_30s: pd.DataFrame, z_threshold: float, trade_min_gap_s: int) -> List[pd.Timestamp]:
    trade_times = df_30s[df_30s["zscore"] > z_threshold].index
    filtered: List[pd.Timestamp] = []
    last_t: Optional[pd.Timestamp] = None

    for t in trade_times:
        if last_t is None or (t - last_t).total_seconds() > trade_min_gap_s:
            filtered.append(t)
            last_t = t
    return filtered


def build_trade_table(
    df_30s: pd.DataFrame,
    filtered_trades: List[pd.Timestamp],
    pre_window_s: int,
    post_window_s: int,
    liq_lookback_s: int,
    min_points: int = 2,
    pip_scale: float = 1e4,
) -> pd.DataFrame:
    rows = []
    dt = 30.0

    for t in filtered_trades:
        if t - pd.Timedelta(seconds=pre_window_s) < df_30s.index.min():
            continue
        if t + pd.Timedelta(seconds=post_window_s) > df_30s.index.max():
            continue
        if t not in df_30s.index:
            continue

        pre = df_30s.loc[t - pd.Timedelta(seconds=pre_window_s): t - pd.Timedelta(seconds=1)]
        post = df_30s.loc[t + pd.Timedelta(seconds=1): t + pd.Timedelta(seconds=post_window_s)]

        if len(pre) < min_points or len(post) < min_points:
            continue

        pre_mid = pre["mid"].to_numpy(dtype=float)
        if np.sum(np.isfinite(pre_mid)) < min_points:
            continue

        mid_t = float(df_30s.at[t, "mid"])
        if not np.isfinite(mid_t):
            continue

        pre_mean = float(np.nanmean(pre_mid))

        skew = float((mid_t - pre_mean) * pip_scale)
        sigma = float(np.nanstd(pre_mid - pre_mean, ddof=1) * pip_scale)
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        volume_proxy = float(df_30s.at[t, "zscore"])
        liq = compute_liq_proxy_from_lookback(df_30s, t, liq_lookback_s)

        post_mid = post["mid"].to_numpy(dtype=float)
        if np.sum(np.isfinite(post_mid)) < min_points:
            continue

        reval_ir = float(((post_mid - mid_t).sum()) * dt * pip_scale)
        reval = float(reval_ir / max(float(post_window_s), 1.0))

        if np.isfinite(skew) and np.isfinite(volume_proxy) and np.isfinite(liq) and np.isfinite(reval):
            rows.append((t, skew, sigma, volume_proxy, liq, reval))

    return pd.DataFrame(
        rows,
        columns=["time", "skew", "sigma", "volume_proxy", "liq", "reval"],
    ).dropna()


def add_target_signal(trades: pd.DataFrame, sigma_f_floor: Optional[float] = None) -> pd.DataFrame:
    trades = trades.copy()

    sigma = trades["sigma"].to_numpy(dtype=float)

    if sigma_f_floor is None:
        sigma_f_floor = float(np.nanquantile(sigma, 0.05))

    sigma_f = np.maximum(sigma, sigma_f_floor)

    side = np.sign(trades["skew"].to_numpy(dtype=float))
    reval = trades["reval"].to_numpy(dtype=float)

    f = side * reval / (sigma_f + 1e-12)
    trades["f"] = f

    return trades