# fx_rates_scatter_with_volatility.py
# EURUSD daily log returns vs Δ(US-EZ 10Y yield differential) [bps], coloured by FX volatility
# Model: r_t = alpha + beta1 * Δi_t + beta2 * sigma_t + eps_t
# Data: Yahoo Finance (EURUSD=X) + FRED (DGS10, IRLTLT01EZM156N)

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

# Volatility definition: rolling std of daily log returns
VOL_WINDOW = 20                  # ~1 month trading days
VOL_ANNUALISE = False            # True -> multiply by sqrt(252)

# Sampling options (to reduce clutter on slides)
USE_SUBSAMPLE = True
SUBSAMPLE_N = 120
SUBSAMPLE_METHOD = "top_abs_x"   # "top_abs_x" or "random"
RANDOM_SEED = 42


# -----------------------------
# Helpers
# -----------------------------
def download_fx_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        else:
            s = df.iloc[:, 0]
    else:
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


def ols_fit_multi(X: np.ndarray, y: np.ndarray):
    """
    OLS: y = X b
    returns (b, r2)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, r2


def choose_points(df: pd.DataFrame, n: int, method: str, seed: int = 0) -> pd.DataFrame:
    if len(df) <= n:
        return df
    if method == "top_abs_x":
        return df.reindex(df["x"].abs().sort_values(ascending=False).index).head(n).sort_index()
    if method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(df.index.values, size=n, replace=False)
        return df.loc[np.sort(idx)]
    return df


# -----------------------------
# Load + build features
# -----------------------------
fx = download_fx_close(FX_TICKER, START, END)
us10 = download_fred_series(US_YIELD, START, END)      # %
ez10 = download_fred_series(EZ_YIELD, START, END)      # %

data = pd.concat([fx, us10, ez10], axis=1).dropna()

# FX daily log returns
data["fx_log_ret"] = np.log(data["fx"]).diff()

# Yield differential (US - EZ) in % points; change in bps => *100
data["yld_diff"] = data[US_YIELD] - data[EZ_YIELD]
data["dyld_diff_bps"] = data["yld_diff"].diff() * 100.0

# Volatility proxy: rolling std of log returns
data["sigma"] = data["fx_log_ret"].rolling(VOL_WINDOW).std()
if VOL_ANNUALISE:
    data["sigma"] = data["sigma"] * np.sqrt(252)

df = data[["dyld_diff_bps", "fx_log_ret", "sigma"]].dropna().rename(
    columns={"dyld_diff_bps": "x", "fx_log_ret": "y", "sigma": "sigma"}
)

# Optional subsampling for cleaner slide look
plot_df = choose_points(df, SUBSAMPLE_N, SUBSAMPLE_METHOD, RANDOM_SEED) if USE_SUBSAMPLE else df

x = plot_df["x"].values
y = plot_df["y"].values
s = plot_df["sigma"].values
N = len(plot_df)

# -----------------------------
# Multiple regression: y = a + b1*x + b2*sigma + eps
# -----------------------------
X = np.column_stack([np.ones_like(x), x, s])
b, r2 = ols_fit_multi(X, y)
a, b1, b2 = b[0], b[1], b[2]

# For a single “macro sensitivity” line on the scatter, fix sigma at its median
sigma_ref = np.median(s)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = a + b1 * x_line + b2 * sigma_ref


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=140)

# Scatter coloured by volatility
sc = ax.scatter(
    x, y,
    c=s,
    s=55,
    alpha=0.75,
    edgecolors="none",
)

# Add colourbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"Volatility $\sigma_t$ (rolling std of returns)", fontsize=20)

# “Partial effect” line (holding sigma fixed at median)
ax.plot(
    x_line, y_line,
    linewidth=6.0,
    alpha=0.95,
    label=rf"Fit holding $\sigma_t$ at median ({sigma_ref:0.4g})",
)

# Zero lines
ax.axhline(0, linewidth=1.4, alpha=0.85)
ax.axvline(0, linewidth=1.4, alpha=0.85)

# Grid
ax.minorticks_on()
ax.grid(True, which="major", alpha=0.28, linewidth=1.0)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.8)

# Labels
ax.set_xlabel(r"$\Delta$ yield differential (US 10Y $-$ EZ proxy), basis points", fontsize=25)
ax.set_ylabel("EURUSD daily log return", fontsize=25)

# Title
title = (
    f"EURUSD daily log returns vs $\\Delta$(US–EZ 10Y yield) [bps]\n"
    f"Points coloured by volatility (rolling {VOL_WINDOW}d std), {START} to {END}"
)
ax.set_title(title, fontsize=30, pad=14)

# Stats box
stats = (
    rf"$\hat{{\beta}}_1$ (rate diff) = {b1:0.6f} per bps" "\n"
    rf"$\hat{{\beta}}_2$ (vol) = {b2:0.6f}" "\n"
    rf"$R^2$ = {r2:0.3f},  $N$ = {N}"
)
ax.text(
    0.02, 0.98, stats,
    transform=ax.transAxes,
    va="top", ha="left",
    fontsize=13,
    bbox=dict(boxstyle="round", alpha=0.15)
)

ax.legend(loc="upper right", framealpha=0.25, fontsize=15)

plt.tight_layout()

out_png = "eurusd_rate_diff_vol_scatter.png"
out_pdf = "eurusd_rate_diff_vol_scatter.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print(f"Saved: {out_png} and {out_pdf}")
print(f"alpha = {a}, beta1 (per bps) = {b1}, beta2 (per sigma unit) = {b2}, R2 = {r2}, N = {N}")