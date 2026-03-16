#resid_analysis.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from diagnostics.plotting import apply_pub_style, robust_limits


def fig_residual_hist(resid, title="Residual distribution (histogram)", xlabel="Residual"):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(resid, bins=40, density=True, alpha=0.75, label="Residuals")

    # Use robust limits also for overlay range (prevents outliers from stretching)
    xlim = robust_limits(resid, lo=0.005, hi=0.995)
    if xlim:
        xs = np.linspace(xlim[0], xlim[1], 300)
        ax.set_xlim(*xlim)
    else:
        xs = np.linspace(np.min(resid), np.max(resid), 300)

    mu = float(np.mean(resid))
    sd = float(np.std(resid) + 1e-12)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), linewidth=2, label="Normal PDF")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    apply_pub_style(ax)
    return fig


def fig_residual_qq(resid, title="Residual Q–Q plot"):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]

    fig = plt.figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(111)

    sm.ProbPlot(resid).qqplot(line="45", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    apply_pub_style(ax)
    return fig