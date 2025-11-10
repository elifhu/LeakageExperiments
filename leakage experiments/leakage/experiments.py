import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import get_fx_data
from analysis.regression_model import fit_ols
from analysis.residual_analysis import plot_residuals
from scipy.stats import mannwhitneyu
import seaborn as sns
import statsmodels.api as sm


# CONFIGURATION

symbol = "EURUSD=X"
start_date = "2025-10-01"
end_date = "2025-11-01"
z_threshold = 2.5
pre_window = 60
post_window = 60
roll = 10
event_min_gap_s = 300


# EXPERIMENT 1: LINEAR REGRESSION

df = get_fx_data(symbol, start=start_date, end=end_date)
df_30s = df.resample("30s").interpolate("linear")
df_30s["dlog"] = np.log(df_30s["mid"] / df_30s["mid"].shift(1))

vol = df_30s["dlog"].rolling(roll, min_periods=roll).std()
df_30s["zscore"] = (df_30s["dlog"] / (vol + 1e-12)).abs()
events = df_30s[df_30s["zscore"] > z_threshold].index

filtered_events, last_t = [], None
for t in events:
    if last_t is None or (t - last_t).total_seconds() > event_min_gap_s:
        filtered_events.append(t)
        last_t = t

rows = []
for t in filtered_events:
    if t - pd.Timedelta(seconds=pre_window) < df_30s.index.min():
        continue
    if t + pd.Timedelta(seconds=post_window) > df_30s.index.max():
        continue

    pre = df_30s.loc[t - pd.Timedelta(seconds=pre_window): t - pd.Timedelta(seconds=1)]
    post = df_30s.loc[t + pd.Timedelta(seconds=1): t + pd.Timedelta(seconds=post_window)]

    if len(pre) < 2 or len(post) < 2:
        continue
    if pre["dlog"].std() == 0 or np.isnan(pre["dlog"].std()):
        continue

    skew = pre["dlog"].mean() / pre["dlog"].std()
    twap_next = post["mid"].mean()
    reval = (twap_next - df_30s.at[t, "mid"]) * 1e4  # convert to bps

    if np.isfinite(skew) and np.isfinite(reval):
        rows.append((t, skew, reval))

ev = pd.DataFrame(rows, columns=["time", "skew", "reval"]).dropna()
print(f"Valid event count: {len(ev)}")

if len(ev) > 3:
    X, Y = ev["skew"], ev["reval"]
    model = fit_ols(X, Y, robust=True)
    print(model.summary())

    plt.figure(figsize=(7, 5))
    plt.scatter(ev["skew"], ev["reval"], alpha=0.5, label="Observed Events", color="gray")
    plt.plot(ev["skew"], model.predict(sm.add_constant(ev["skew"])),
             color="red", linewidth=2, label="Fitted Regression Line")
    plt.title("Relationship Between Skew and Integrated Reval (bps) (EUR/USD, Oct-Nov 2025)", fontsize=12, pad=10)
    plt.xlabel("Skew (Directional Bias)", fontsize=11)
    plt.ylabel("Integrated Reval (bps)", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plot_residuals(model.resid)


# EXPERIMENT 2: MANN–WHITNEY U TEST

print("\n=== Experiment 2: Nonparametric Distributional Test ===")

pos_reval = ev[ev["skew"] > 0]["reval"].dropna()
neg_reval = ev[ev["skew"] < 0]["reval"].dropna()

U, p_value = mannwhitneyu(pos_reval, neg_reval, alternative="two-sided")

# Combine for plotting
plot_df = pd.DataFrame({
    "Reval (bps)": np.concatenate([pos_reval.values, neg_reval.values]),
    "Skew Regime": (["Positive Skew"] * len(pos_reval)) + (["Negative Skew"] * len(neg_reval))
})

# Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=plot_df,
    x="Skew Regime",
    y="Reval (bps)",
    palette=["#3B9AE1", "#E15759"],
    inner="box",
    cut=0,
    linewidth=1
)
plt.title("Distribution of Reval Across Skew Regimes (EUR/USD, Oct-Nov 2025)", fontsize=12, pad=10)
plt.xlabel("Skew Regime", fontsize=11)
plt.ylabel("Integrated Reval (bps)", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Result summary
print("\nMann–Whitney U Test Results:")
print(f"U statistic = {U:.2f}")
print(f"p-value     = {p_value:.4f}")
print(f"Median (positive skew) = {pos_reval.median():.4f} bps")
print(f"Median (negative skew) = {neg_reval.median():.4f} bps")

if p_value < 0.05:
    print("→ Statistically significant difference between regimes (possible information leakage).")
else:
    print("→ No significant difference between regimes (no evidence of leakage).")