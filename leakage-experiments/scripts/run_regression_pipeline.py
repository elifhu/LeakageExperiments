from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

from paths import TABLES_DIR, ensure_dirs
from leakage.dataset_builder import load_trade_dataset


ALPHA_TOX = 0.05
MIN_SAMPLES_PER_SIDE = 5
N_SYNTH_COUNTERPARTIES = 12
RANDOM_SEED = 42

OUT_TAB = TABLES_DIR / "table_state_regression.tex"
OUT_CP_BASE = TABLES_DIR / "table_counterparty_toxicity_baseline.tex"


def attach_synthetic_counterparties(trades: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    out = trades.copy()
    out["counterparty_id"] = rng.integers(0, N_SYNTH_COUNTERPARTIES, size=len(out))

    side = np.where(out["skew"] >= 0, 1, -1)
    flip_mask = rng.random(len(out)) < 0.15
    side[flip_mask] *= -1

    out["side"] = side
    out["signed_reval"] = out["side"] * out["reval"]
    return out


def compute_counterparty_toxicity(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cp, g in trades.groupby("counterparty_id"):
        pos = g.loc[g["skew"] > 0, "signed_reval"].to_numpy()
        neg = g.loc[g["skew"] < 0, "signed_reval"].to_numpy()

        n_pos = len(pos)
        n_neg = len(neg)

        U = np.nan
        p = np.nan
        delta = np.nan
        toxicity_base = np.nan
        valid = False

        if n_pos >= MIN_SAMPLES_PER_SIDE and n_neg >= MIN_SAMPLES_PER_SIDE:
            valid = True
            U, p = mannwhitneyu(pos, neg, alternative="two-sided")
            delta = float(np.median(pos) - np.median(neg))
            toxicity_base = delta if p < ALPHA_TOX else 0.0

        rows.append(
            {
                "counterparty_id": cp,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "U": U,
                "p_value": p,
                "delta_median": delta,
                "toxicity_base": toxicity_base,
                "valid_test": valid,
            }
        )

    out = pd.DataFrame(rows)
    out["abs_toxicity"] = np.abs(out["toxicity_base"])
    return out.sort_values(by=["valid_test", "abs_toxicity"], ascending=[False, False])


def export_regression_table(model, path) -> None:
    names = {
        "const": "Intercept",
        "Skew": "Skew",
        "Volatility": "Volatility",
        "Volume proxy": "Volume proxy",
        "Liquidity proxy": "Liquidity proxy",
    }

    rows = []
    for name in model.params.index:
        rows.append(
            {
                "Variable": names.get(name, name),
                "Coef.": model.params[name],
                "HC3 SE": model.bse[name],
                "z": model.tvalues[name],
                "p-value": model.pvalues[name],
            }
        )

    df = pd.DataFrame(rows)

    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Variable & Coef. & HC3 SE & z & p-value \\\\\n")
        f.write("\\midrule\n")

        for _, r in df.iterrows():
            f.write(
                f"{r['Variable']} & "
                f"{r['Coef.']:.4f} & "
                f"{r['HC3 SE']:.4f} & "
                f"{r['z']:.2f} & "
                f"{r['p-value']:.3g} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{State-dependent regression of reval on skew, volatility, "
            "volume proxy and liquidity proxy with HC3 robust standard errors.}\n"
        )
        f.write("\\label{tab:state_regression}\n")
        f.write("\\end{table}\n")


def export_counterparty_table(df: pd.DataFrame, path) -> None:
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("CP & $n_{+}$ & $n_{-}$ & $U$ & $p$ & $\\Delta$ & Toxicity \\\\\n")
        f.write("\\midrule\n")

        for _, r in df.iterrows():
            def fmt(x, nd=3):
                return "--" if pd.isna(x) else f"{x:.{nd}f}"

            f.write(
                f"{int(r['counterparty_id'])} & "
                f"{int(r['n_pos'])} & "
                f"{int(r['n_neg'])} & "
                f"{fmt(r['U'], 1)} & "
                f"{fmt(r['p_value'], 3)} & "
                f"{fmt(r['delta_median'], 3)} & "
                f"{fmt(r['toxicity_base'], 3)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Baseline counterparty toxicity summary obtained from "
            "signed reval differences across positive- and negative-skew trades.}\n"
        )
        f.write("\\label{tab:counterparty_toxicity_baseline}\n")
        f.write("\\end{table}\n")


def main():
    ensure_dirs()

    trades = load_trade_dataset("trade_table")
    print("Trades loaded:", len(trades))

    pos = trades.loc[trades["skew"] > 0, "reval"]
    neg = trades.loc[trades["skew"] < 0, "reval"]

    U, p = mannwhitneyu(pos, neg, alternative="two-sided")
    print(f"Global Mann-Whitney test: U={U:.2f}, p={p:.6f}")

    X = trades[["skew", "sigma", "volume_proxy", "liq"]]

    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=["Skew", "Volatility", "Volume proxy", "Liquidity proxy"],
        index=trades.index,
    )

    X = sm.add_constant(X)
    Y = trades["reval"]

    model = sm.OLS(Y, X).fit(cov_type="HC3")
    print(model.summary())

    export_regression_table(model, OUT_TAB)
    print("Saved regression table:", OUT_TAB)

    trades_cp = attach_synthetic_counterparties(trades)
    cp = compute_counterparty_toxicity(trades_cp)
    export_counterparty_table(cp, OUT_CP_BASE)
    print("Saved counterparty toxicity:", OUT_CP_BASE)


if __name__ == "__main__":
    main()