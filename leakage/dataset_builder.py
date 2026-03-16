from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_CONFIG, TRADE_CONFIG
from paths import TRADES_DIR, ensure_dirs
from utils.data_loader import get_fx_data
from leakage.leakage_events import detect_trades, build_trade_table, add_target_signal


CANONICAL_TRADE_COLUMNS = [
    "trade_id",
    "time",
    "skew",
    "sigma",
    "volume_proxy",
    "liq",
    "reval",
    "f",
]


def last_nd_dates(n_days: int) -> tuple[str, str]:
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=n_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_resampled_fx_frame(
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str | None = None,
    roll: int | None = None,
) -> pd.DataFrame:
    symbol = symbol or DATA_CONFIG.symbol
    interval = interval or DATA_CONFIG.interval
    roll = roll or TRADE_CONFIG.roll

    if start_date is None or end_date is None:
        start_date, end_date = last_nd_dates(DATA_CONFIG.lookback_days)

    df = get_fx_data(symbol, start=start_date, end=end_date, interval=interval)
    df_30s = df.resample("30s").interpolate("linear")

    df_30s["dlog"] = np.log(df_30s["mid"] / df_30s["mid"].shift(1))
    vol = df_30s["dlog"].rolling(roll, min_periods=roll).std()
    df_30s["zscore"] = (df_30s["dlog"] / (vol + 1e-12)).abs()

    return df_30s


def build_trade_dataset(
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    df_30s = build_resampled_fx_frame(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    filtered_trades = detect_trades(
        df_30s,
        z_threshold=TRADE_CONFIG.z_threshold,
        trade_min_gap_s=TRADE_CONFIG.trade_min_gap_s,
    )

    trades = build_trade_table(
        df_30s=df_30s,
        filtered_trades=filtered_trades,
        pre_window_s=TRADE_CONFIG.pre_window_s,
        post_window_s=TRADE_CONFIG.post_window_s,
        liq_lookback_s=TRADE_CONFIG.liq_lookback_s,
        min_points=TRADE_CONFIG.min_points,
        pip_scale=TRADE_CONFIG.pip_scale,
    )

    trades = add_target_signal(trades)

    required = ["time", "skew", "sigma", "volume_proxy", "liq", "reval", "f"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"Missing required trade columns after build: {missing}")

    trades = trades[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
    trades = trades.dropna(subset=["time"]).reset_index(drop=True)
    trades.insert(0, "trade_id", np.arange(len(trades), dtype=int))

    return trades


def save_trade_dataset(trades: pd.DataFrame, stem: str = "trade_table") -> tuple[Path, Path]:
    ensure_dirs()

    csv_path = TRADES_DIR / f"{stem}.csv"
    parquet_path = TRADES_DIR / f"{stem}.parquet"

    trades.to_csv(csv_path, index=False)
    trades.to_parquet(parquet_path, index=False)

    return csv_path, parquet_path


def load_trade_dataset(stem: str = "trade_table") -> pd.DataFrame:
    parquet_path = TRADES_DIR / f"{stem}.parquet"
    csv_path = TRADES_DIR / f"{stem}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["time"])

    raise FileNotFoundError(
        f"Could not find trade dataset at {parquet_path} or {csv_path}"
    )