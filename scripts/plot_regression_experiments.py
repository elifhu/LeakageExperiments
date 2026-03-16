from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu
import statsmodels.api as sm

from paths import FIGURES_DIR, ensure_dirs
from leakage.dataset_builder import load_trade_dataset


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11.5,
})

COL_BLUE = "#355C8A"
COL_ORANGE = "#B45A30"
COL_GRID = "#E6E6E6"


def load_baseline_data():
    trades = load_trade_dataset("trade_table")

    if len(trades) == 0:
        raise ValueError("Trade table is empty.")

    skew = trades["skew"].to_numpy(dtype=float)
    reval = trades["reval"].to_numpy(dtype=float)

    mask = np.isfinite(skew) & np.isfinite(reval)
    skew = skew[mask]
    reval = reval[mask]

    x = sm.add_constant(skew)
    model = sm.OLS(reval, x).fit(cov_type="HC3")

    pos = reval[skew > 0]
    neg = reval[skew < 0]

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Need both positive- and negative-skew trades.")

    u_stat, p_val = mannwhitneyu(pos, neg, alternative="two-sided")

    return skew, reval, model, pos, neg, u_stat, p_val


def style_axis(ax):
    ax.grid(color=COL_GRID, linewidth=0.6, alpha=0.9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=3.5, width=0.9)


def add_regression_panel(ax, skew, reval, model):
    ax.scatter(
        skew,
        reval,
        s=18,
        alpha=0.30,
        color=COL_BLUE,
        edgecolors="none",
        rasterized=True,
    )

    xs = np.linspace(np.min(skew), np.max(skew), 300)
    ys = model.predict(sm.add_constant(xs))
    ax.plot(xs, ys, linewidth=2.2, color=COL_ORANGE)

    ax.set_xlabel("Directional skew")
    ax.set_ylabel("Integrated reval")
    ax.set_title("Regression baseline", pad=10, fontsize=12.5)

    style_axis(ax)


def add_density_panel(ax, pos, neg):
    xmin = min(pos.min(), neg.min())
    xmax = max(pos.max(), neg.max())
    pad = 0.08 * (xmax - xmin)

    xs = np.linspace(xmin - pad, xmax + pad, 500)

    kde_pos = gaussian_kde(pos)
    kde_neg = gaussian_kde(neg)

    yp = kde_pos(xs)
    yn = kde_neg(xs)

    ax.plot(xs, yp, color=COL_BLUE, linewidth=2.0, label=r"skew > 0")
    ax.plot(xs, yn, color=COL_ORANGE, linewidth=2.0, label=r"skew < 0")

    ax.axvline(np.median(pos), color=COL_BLUE, linestyle="--", linewidth=1.1, alpha=0.75)
    ax.axvline(np.median(neg), color=COL_ORANGE, linestyle="--", linewidth=1.1, alpha=0.75)

    ax.set_xlabel("Integrated reval")
    ax.set_ylabel("Density")
    ax.set_title("Integrated reval density conditional on skew", pad=10, fontsize=12.5)

    style_axis(ax)
    ax.legend(
        frameon=False,
        fontsize=11.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.92),
        handlelength=2.4,
    )


def make_baseline_diagnostics_figure(out_path: str):
    skew, reval, model, pos, neg, u_stat, p_val = load_baseline_data()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7.2, 8.6),
        constrained_layout=True,
    )

    add_regression_panel(axes[0], skew, reval, model)
    add_density_panel(axes[1], pos, neg)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return model, u_stat, p_val


def main():
    ensure_dirs()
    out = os.path.join(FIGURES_DIR, "fig_baseline_diagnostics.pdf")
    model, u_stat, p_val = make_baseline_diagnostics_figure(out)

    print("Saved:", out)
    print(model.summary())
    print(f"Global Mann-Whitney U = {u_stat:.3f}, p = {p_val:.6f}")


if __name__ == "__main__":
    main()