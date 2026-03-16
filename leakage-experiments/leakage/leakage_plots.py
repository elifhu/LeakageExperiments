#leakage_plots.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def plot_geometry_with_nodecolor(
    G,
    pos,
    node_color,
    title,
    cbar_label,
    node_size=45,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.10,
        width=0.5,
        ax=ax
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        cmap="viridis",
        linewidths=0.0,
        ax=ax
    )

    fig.colorbar(nodes, fraction=0.046, pad=0.02, label=cbar_label, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_spectral_embedding(
    x,
    y,
    c,
    title="Spectral Embedding of Trade Similarity Graph",
    xlabel="Graph Fourier coordinate 1",
    ylabel="Graph Fourier coordinate 2",
    cbar_label="Structural anomaly score",
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(
        x, y,
        c=c,
        s=18,
        alpha=0.85,
        cmap="coolwarm",
        linewidths=0.0
    )
    fig.colorbar(sc, fraction=0.046, pad=0.02, label=cbar_label, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_zlocal_over_time_highlight(
    times,
    z_local,
    threshold=2.0,
    title="Structural Anomaly Score Over Trade Time",
):
    t = pd.to_datetime(times)
    z = np.asarray(z_local, dtype=float)

    order = np.argsort(t.values)
    t = t.values[order]
    z = z[order]

    normal = z <= threshold
    anomalous = z > threshold

    z_series = pd.Series(z)
    z_med = z_series.rolling(
        25, min_periods=5
    ).median().values

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.scatter(
        t[normal], z[normal],
        s=10, alpha=0.35,
        label="Trades"
    )
    ax.scatter(
        t[anomalous], z[anomalous],
        s=18, alpha=0.85,
        label="Anomalous trades"
    )

    ax.plot(t, z_med, linewidth=2)

    ax.axhline(threshold, linestyle="--")
    ax.axhline(-threshold, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Trade time")
    ax.set_ylabel("Local smoothness anomaly (z-score)")
    ax.legend(frameon=False)
    plt.xticks(rotation=25)
    fig.tight_layout()
    return fig