from __future__ import annotations

import numpy as np
import pandas as pd


def assign_synthetic_counterparties_continuous(
    trades: pd.DataFrame,
    n_counterparties: int = 80,
    tail_df: float = 3.0,
    tail_scale: float = 1.0,
    beta: float = 1.2,
    base_mix: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = trades.copy()

    counterparties = np.arange(n_counterparties, dtype=int)

    theta_all = rng.standard_t(df=tail_df, size=n_counterparties) * tail_scale
    w = base_mix + np.exp(beta * np.abs(theta_all))
    p = w / w.sum()

    counterparty_id = rng.choice(counterparties, size=len(out), replace=True, p=p)
    out["counterparty_id"] = counterparty_id
    out["theta"] = theta_all[counterparty_id]

    return out


def make_leakage_target_f(
    trades: pd.DataFrame,
    alpha: float = 1.0,
    noise: float = 0.25,
    p_align: float = 0.7,
    seed: int = 123,
    sigma_col: str = "sigma",
    skew_col: str = "skew",
    reval_col: str = "reval",
    theta_col: str = "theta",
    out_col: str = "f",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = trades.copy()

    eps = 1e-12
    sigma = out[sigma_col].to_numpy(dtype=float)
    reval = out[reval_col].to_numpy(dtype=float)
    side = np.sign(out[skew_col].to_numpy(dtype=float))
    theta = out[theta_col].to_numpy(dtype=float)

    f_real = side * reval / (sigma + eps)
    sig_scale = sigma / (np.nanmedian(sigma) + eps)
    align = np.where(rng.random(len(out)) < p_align, 1.0, -1.0)
    leakage = align * theta * sig_scale

    out[out_col] = f_real + alpha * leakage + noise * rng.standard_normal(len(out))
    return out