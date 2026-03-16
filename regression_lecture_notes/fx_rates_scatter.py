# fx_rates_scatter_final.py
# EURUSD daily log returns vs Δ(US-EZ 10Y yield differential) [bps]
# Real data: Yahoo Finance (EURUSD=X) + FRED (DGS10, IRLTLT01EZM156N)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr


# -----------------------------
# Config
# -----------------------------
START = "2018-05-01"
END   = "2024-10-01"

FX_TICKER = "EURUSD=X"

# FRED series:
US_YIELD  = "DGS10"              # 10-Year Treasury Constant Maturity Rate (%)
EZ_YIELD  = "IRLTLT01EZM156N"    # Euro Area Long-Term Interest Rate (proxy, %)

# Sampling options (to reduce clutter like your prof's slides)
USE_SUBSAMPLE = True
SUBSAMPLE_N = 80          # number of points to show if subsampling
SUBSAMPLE_METHOD = "top_abs_x"   # "top_abs_x" or "random"

RANDOM_SEED = 42

# -----------------------------
# Helpers
# -----------------------------
def download_fx_close(ticker: str, start: str, end: str) -> pd.Series:
    """
    yfinance now often returns a MultiIndex columns df.
    We'll robustly pick 'Close' (or 'Adj Close' if needed).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # If columns are MultiIndex, select the ticker sub-column
    if isinstance(df.columns, pd.MultiIndex):
        # preferred: Close
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        else:
            # fallback: first column
            s = df.iloc[:, 0]
    else:
        # Single level columns
        if "Close" in df.columns:
            s = df["Close"]
        elif "Adj Close" in df.columns:
            s = df["Adj Close"]
        else:
            s = df.iloc[:, 0]

    s = s.dropna()
    s.name = "fx"
    return s


def download_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    s = pdr.DataReader(series_id, "fred", start, end)[series_id].dropna()
    s.name = series_id
    return s


def ols_fit(x: np.ndarray, y: np.ndarray):
    """
    Simple OLS y = a + b x, returns (a, b, r2)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a, b = beta[0], beta[1]

    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def choose_points(df: pd.DataFrame, n: int, method: str, seed: int = 0) -> pd.DataFrame:
    """
    Reduce number of scatter points to make the slide cleaner.
    method:
      - "top_abs_x": choose days with largest |x| (largest rate diff moves)
      - "random": random sample
    """
    if len(df) <= n:
        return df

    if method == "top_abs_x":
        return df.reindex(df["x"].abs().sort_values(ascending=False).index).head(n).sort_index()
    elif method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(df.index.values, size=n, replace=False)
        return df.loc[np.sort(idx)]
    else:
        return df


# -----------------------------
# Load + build features
# -----------------------------
fx = download_fx_close(FX_TICKER, START, END)
us10 = download_fred_series(US_YIELD, START, END)      # %
ez10 = download_fred_series(EZ_YIELD, START, END)      # %

# Align on common dates
data = pd.concat([fx, us10, ez10], axis=1).dropna()

# FX daily log returns
data["fx_log_ret"] = np.log(data["fx"]).diff()

# Yield differential (US - EZ) in % points; change in bps => *100
data["yld_diff"] = data[US_YIELD] - data[EZ_YIELD]
data["dyld_diff_bps"] = data["yld_diff"].diff() * 100.0  # 1% = 100 bps

# Keep clean rows
df = data[["dyld_diff_bps", "fx_log_ret"]].dropna().rename(
    columns={"dyld_diff_bps": "x", "fx_log_ret": "y"}
)



# Optional subsampling for cleaner “lecture slide” look
plot_df = choose_points(df, SUBSAMPLE_N, SUBSAMPLE_METHOD, RANDOM_SEED) if USE_SUBSAMPLE else df

x = plot_df["x"].values
y = plot_df["y"].values

a, b, r2 = ols_fit(x, y)
N = len(plot_df)

# Line for fit
x_line = np.linspace(x.min(), x.max(), 200)
y_line = a + b * x_line


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=140)

# Scatter
ax.scatter(
    x, y,
    s=55,
    alpha=0.6,
    color='navy',
    edgecolors="none",
    label="Daily observations"
)

# OLS fit
ax.plot(
    x_line, y_line,
    linewidth=7.0,
    alpha=0.95,
    label="Regression line",
    color='red'
)

# Zero lines (clean “crosshair” look)
ax.axhline(0, linewidth=1.4, alpha=0.85)
ax.axvline(0, linewidth=1.4, alpha=0.85)

# Grid styling (major + minor)
ax.minorticks_on()
ax.grid(True, which="major", alpha=0.28, linewidth=1.0)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.8)

# Labels
ax.set_xlabel(r"$\Delta$ yield differential (US 10Y $-$ EZ proxy), basis points", fontsize=25)
ax.set_ylabel("EURUSD daily log return", fontsize=25)

# Title with dates
title = f"EURUSD daily log returns vs $\\Delta$(US–EZ 10Y yield) [bps]\n{START} to {END}"
ax.set_title(title, fontsize=35, pad=16)

# Stats box (β per bps)
stats = (
    rf"$\hat{{\beta}}$ = {b:0.6f} per bps" "\n"
    rf"$R^2$ = {r2:0.3f}" "\n"
)
ax.text(
    0.03, 0.93, stats,
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=18,
)

# Legend (top-right)
leg = ax.legend(loc="upper right", framealpha=0.25, fontsize=12)

# Tight layout
plt.tight_layout()

# Save (nice for LaTeX)
out_png = "eurusd_vs_rate_diff_scatter.png"
out_pdf = "eurusd_vs_rate_diff_scatter.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print(f"Saved: {out_png} and {out_pdf}")
print(f"beta (per bps) = {b}, R2 = {r2}, N = {N}")