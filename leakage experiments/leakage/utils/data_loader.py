import yfinance as yf
import pandas as pd
from datetime import timedelta


def get_fx_data(symbol="EURUSD=X", start="2025-10-01", end="2025-11-01"):
    """
    Download high-frequency FX mid-price data (1-minute) from Yahoo Finance.
    Yahoo only allows 7 days of 1m data per request — data is fetched in consecutive chunks.
    Returns a continuous and cleaned mid-price time series.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    data_list = []

    while start < end:
        chunk_end = min(start + timedelta(days=7), end)
        d = yf.download(symbol, start=start, end=chunk_end, interval="1m", progress=False)[["Close"]]
        if not d.empty:
            data_list.append(d)
        start = chunk_end

    df = pd.concat(data_list)
    df = df[~df.index.duplicated()].sort_index()
    df.columns = ["mid"]

    # Save locally for reproducibility
    df.to_csv(f"data/{symbol.replace('=','_')}_1m_{start.date()}_{end.date()}.csv")
    print(f"Downloaded {len(df)} rows for {symbol} between {df.index.min()} and {df.index.max()}.")
    return df