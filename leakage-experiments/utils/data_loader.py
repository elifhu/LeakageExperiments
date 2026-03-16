from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import timedelta
from pathlib import Path

from paths import DATA_DIR


def _cache_path(symbol: str, interval: str, start, end) -> Path:
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    name = f"{symbol.replace('=', '_')}_{interval}_{start}_{end}.csv"
    return DATA_DIR / name


def get_fx_data(
    symbol: str,
    start,
    end,
    interval: str = "1m",
    save: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    orig_start = pd.to_datetime(start)
    orig_end = pd.to_datetime(end)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(symbol, interval, orig_start, orig_end)

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df = df.sort_index()
        if "mid" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "mid"})
        print(f"Loaded cache: {cache_file} ({len(df)} rows)")
        return df[["mid"]]

    cur = orig_start
    data_list = []

    while cur < orig_end:
        chunk_end = min(cur + timedelta(days=7), orig_end)

        d = yf.download(
            symbol,
            start=cur,
            end=chunk_end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if not d.empty and "Close" in d.columns:
            data_list.append(d[["Close"]])

        cur = chunk_end

    if not data_list:
        raise RuntimeError(
            f"No data returned for {symbol} with interval={interval} "
            f"between {orig_start} and {orig_end}."
        )

    df = pd.concat(data_list)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.columns = ["mid"]

    if save:
        df.to_csv(cache_file)
        print(f"Saved: {cache_file}")

    print(f"Downloaded {len(df)} rows for {symbol} between {df.index.min()} and {df.index.max()}.")
    return df