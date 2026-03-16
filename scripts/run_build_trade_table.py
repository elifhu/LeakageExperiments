from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DATA_CONFIG
from paths import ensure_dirs
from leakage.dataset_builder import build_trade_dataset, save_trade_dataset, last_nd_dates


def main() -> None:
    ensure_dirs()

    start_date, end_date = last_nd_dates(DATA_CONFIG.lookback_days)

    trades = build_trade_dataset(
        symbol=DATA_CONFIG.symbol,
        start_date=start_date,
        end_date=end_date,
    )

    csv_path, parquet_path = save_trade_dataset(trades, stem="trade_table")

    print(f"Built trade dataset with {len(trades)} rows")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print("Columns:", list(trades.columns))


if __name__ == "__main__":
    main()