#plotting.py
from __future__ import annotations

import os
import numpy as np


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def robust_limits(x, lo=0.005, hi=0.995, pad=0.05):
    """
    Quantile-based axis limits to avoid outliers dominating the plot.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None

    a, b = np.quantile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    if b <= a:
        m = float(np.median(x))
        s = float(np.std(x) + 1e-12)
        return (m - 3 * s, m + 3 * s)

    rng = b - a
    return (a - pad * rng, b + pad * rng)


def apply_pub_style(ax):
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    # cleaner look
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def save_pdf(fig, outpath_pdf: str, dpi=300):
    ensure_dir(os.path.dirname(outpath_pdf))
    fig.tight_layout()
    fig.savefig(outpath_pdf, dpi=dpi, bbox_inches="tight")