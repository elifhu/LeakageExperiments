#make_spectral_gsp_flow_figure.py
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# TYPOGRAPHY / STYLE

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
})

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(7)

COL_CLIENT = "#233B5D"
COL_TRADE = "#E6862A"
COL_EDGE_TT = "#C8D0DA"
COL_EDGE_CT = "#8FA7C3"
COL_TEXT = "#111111"
COL_ARROW = "#5E6B7A"


# DEMO DATA

@dataclass
class DemoData:
    trade_xy: np.ndarray
    W: np.ndarray
    y_raw: np.ndarray
    y_lp: np.ndarray
    y_hp: np.ndarray
    local_energy: np.ndarray
    client_ids: np.ndarray
    theta_lp: np.ndarray
    theta_hp: np.ndarray
    theta_p: np.ndarray


def make_demo_data(
    n_trades: int = 150,
    n_clients: int = 8,
    k_nn: int = 8,
    t_diff: float = 4.0,
) -> DemoData:
    centers = np.array([
        [-1.5, 1.0],
        [0.2, 1.5],
        [1.7, 0.2],
        [-0.3, -1.4],
        [1.6, -1.2],
    ])

    pts = []
    for c in centers:
        pts.append(c + 0.35 * np.random.randn(30, 2))
    trade_xy = np.vstack(pts)

    pseudo_t = np.linspace(0, 1, len(trade_xy))[:, None]
    d_space = cdist(trade_xy, trade_xy, metric="sqeuclidean")
    d_time = cdist(pseudo_t, pseudo_t, metric="sqeuclidean")

    K_state = np.exp(-d_space)
    K_time = np.exp(-d_time)

    W = K_state * K_time
    np.fill_diagonal(W, 0.0)

    mask = np.zeros_like(W, dtype=bool)
    for i in range(len(W)):
        idx = np.argsort(-W[i])[:k_nn]
        mask[i, idx] = True
    mask = np.logical_or(mask, mask.T)
    W = W * mask

    deg = W.sum(1)
    D_inv = np.diag(1.0 / np.sqrt(deg + 1e-12))
    L = np.eye(len(W)) - D_inv @ W @ D_inv

    base = 0.8 * np.sin(2.5 * trade_xy[:, 0]) + 0.5 * np.cos(2.2 * trade_xy[:, 1])

    client_ids = np.random.choice(np.arange(n_clients), size=len(trade_xy), replace=True)
    client_style = 0.8 * np.random.randn(n_clients)
    spikes = client_style[client_ids] * (0.4 + 0.6 * np.random.rand(len(trade_xy)))

    y_raw = base + 0.55 * spikes + 0.22 * np.random.randn(len(base))

    L_sp = csr_matrix(L)
    y_lp = expm_multiply(-t_diff * L_sp, y_raw)
    y_hp = y_raw - y_lp

    local_energy = np.sum(W * (y_raw[:, None] - y_raw[None, :]) ** 2, axis=1)

    theta_lp = np.zeros(n_clients)
    theta_hp = np.zeros(n_clients)
    theta_p = np.zeros(n_clients)

    for c in range(n_clients):
        idx = np.where(client_ids == c)[0]
        theta_lp[c] = np.median(y_lp[idx])
        theta_hp[c] = np.median(np.abs(y_hp[idx]))
        theta_p[c] = np.median(local_energy[idx])

    return DemoData(
        trade_xy=trade_xy,
        W=W,
        y_raw=y_raw,
        y_lp=y_lp,
        y_hp=y_hp,
        local_energy=local_energy,
        client_ids=client_ids,
        theta_lp=theta_lp,
        theta_hp=theta_hp,
        theta_p=theta_p,
    )


# ============================================================
# HELPERS
# ============================================================

def add_panel(ax, x, y, w, h, lw=1.25):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=lw,
        facecolor="white",
        edgecolor="black",
        zorder=1,
    )
    ax.add_patch(p)


def add_arrow(ax, x1, y1, x2, y2, ms=18):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=1.5,
        edgecolor=COL_ARROW,
        facecolor=COL_ARROW,
        zorder=20,
        connectionstyle="arc3",
    )
    ax.add_patch(arr)


def panel_title(ax, x, y, w, h, text, fontsize=10.8):
    ax.text(
        x + w / 2,
        y + h - 0.040,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COL_TEXT,
        zorder=10,
    )


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_trade_similarity_graph(ax, pts, W):
    G = nx.Graph()
    for i in range(len(pts)):
        G.add_node(i)

    ii, jj = np.where(np.triu(W > 0, 1))
    for i, j in zip(ii, jj):
        G.add_edge(i, j)

    pos = {i: (pts[i, 0], pts[i, 1]) for i in range(len(pts))}

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.42,
        width=0.75,
        edge_color=COL_EDGE_TT,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=12,
        node_color=COL_TRADE,
        linewidths=0,
    )
    clean_axis(ax)


def draw_hetero_graph(ax, n_clients_show=6, n_trades_show=18):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    client_y = np.linspace(0.82, 0.22, n_clients_show)
    cx = 0.18

    n_left = n_trades_show // 2
    n_right = n_trades_show - n_left

    trade_y_left = np.linspace(0.86, 0.12, n_left)
    trade_y_right = np.linspace(0.82, 0.16, n_right)

    tx_left = 0.72
    tx_right = 0.84

    for yy in client_y:
        ax.scatter(cx, yy, s=72, color=COL_CLIENT, zorder=4)

    for yy in trade_y_left:
        ax.scatter(tx_left, yy, s=22, color=COL_TRADE, zorder=4)
    for yy in trade_y_right:
        ax.scatter(tx_right, yy, s=22, color=COL_TRADE, zorder=4)

    trade_positions = [(tx_left, yy) for yy in trade_y_left] + [(tx_right, yy) for yy in trade_y_right]

    rng = np.random.default_rng(11)
    for yy in client_y:
        dest = rng.choice(np.arange(len(trade_positions)), size=rng.integers(3, 6), replace=False)
        for j in dest:
            tx, ty = trade_positions[j]
            ax.plot(
                [cx + 0.03, tx - 0.02],
                [yy, ty],
                color=COL_EDGE_CT,
                linewidth=0.95,
                alpha=0.72,
                zorder=2,
            )

    tt_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (8, 9), (9, 10),
        (0, 10), (2, 12), (4, 14), (7, 15),
        (11, 16), (13, 17)
    ]
    for i, j in tt_pairs:
        if i < len(trade_positions) and j < len(trade_positions):
            x1, y1 = trade_positions[i]
            x2, y2 = trade_positions[j]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=COL_EDGE_TT,
                linewidth=0.9,
                alpha=0.6,
                zorder=1,
            )

    ax.text(cx, 0.95, "Counterparties", ha="center", fontsize=8.8)
    ax.text((tx_left + tx_right) / 2, 0.95, "Trades", ha="center", fontsize=8.8)


def draw_block_matrix(ax, x_center, y_center, scale=1.0):
    bw = 0.064 * scale
    bh = 0.068 * scale

    x0 = x_center - bw / 2
    x1 = x_center + bw / 2
    y0 = y_center - bh / 2
    y1 = y_center + bh / 2

    ax.plot([x0, x0], [y0, y1], color="black", lw=1.0, zorder=10)
    ax.plot([x0, x0 + 0.006 * scale], [y1, y1], color="black", lw=1.0, zorder=10)
    ax.plot([x0, x0 + 0.006 * scale], [y0, y0], color="black", lw=1.0, zorder=10)

    ax.plot([x1, x1], [y0, y1], color="black", lw=1.0, zorder=10)
    ax.plot([x1 - 0.006 * scale, x1], [y1, y1], color="black", lw=1.0, zorder=10)
    ax.plot([x1 - 0.006 * scale, x1], [y0, y0], color="black", lw=1.0, zorder=10)

    xc1 = x0 + 0.018 * scale
    xc2 = x0 + 0.046 * scale
    yt = y_center + 0.017 * scale
    yb = y_center - 0.017 * scale

    ax.text(xc1, yt, r"$0$", ha="center", va="center", fontsize=10.5)
    ax.text(xc2, yt, r"$\mathbf{B}$", ha="center", va="center", fontsize=10.5)
    ax.text(xc1, yb, r"$\mathbf{B}^{\top}$", ha="center", va="center", fontsize=10.5)
    ax.text(xc2, yb, r"$\mathbf{W}$", ha="center", va="center", fontsize=10.5)


def draw_toxicity_vector_panel(ax, theta_lp, theta_hp, theta_p, client_names=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(theta_lp)
    if client_names is None:
        client_names = [f"CP{i}" for i in range(n)]

    def scale(v):
        v = np.asarray(v, dtype=float)
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    s_lp = scale(np.abs(theta_lp))
    s_hp = scale(theta_hp)
    s_p = scale(theta_p)

    y_positions = np.linspace(0.82, 0.16, n)

    ax.text(0.42, 0.93, r"$S_c$", ha="center", va="bottom", fontsize=9.5)
    ax.text(0.62, 0.93, r"$R_c$", ha="center", va="bottom", fontsize=9.5)
    ax.text(0.82, 0.93, r"$P_c$", ha="center", va="bottom", fontsize=9.5)

    for i, yy in enumerate(y_positions):
        ax.text(0.08, yy, client_names[i], ha="left", va="center", fontsize=8)

        vals = [s_lp[i], s_hp[i], s_p[i]]
        xs = [0.42, 0.62, 0.82]
        cols = ["#4C78A8", "#F58518", "#54A24B"]

        for x, val, col in zip(xs, vals, cols):
            ax.add_patch(FancyBboxPatch(
                (x - 0.055, yy - 0.016),
                0.105 * val + 0.002,
                0.032,
                boxstyle="round,pad=0.002,rounding_size=0.005",
                linewidth=0.35,
                edgecolor="black",
                facecolor=col,
                alpha=0.88,
            ))


def draw_manifold_surface(fig, pos):
    ax3d = fig.add_axes(pos, projection="3d")

    u = np.linspace(-1.15, 1.15, 44)
    v = np.linspace(-1.0, 1.0, 44)
    U, V = np.meshgrid(u, v)

    Z = 0.48 * (U**2 - 0.7 * V**2) + 0.10 * np.sin(2.1 * U) * np.cos(1.7 * V)

    ax3d.plot_surface(
        U, V, Z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )

    ax3d.view_init(elev=23, azim=-58)
    ax3d.set_axis_off()
    ax3d.set_box_aspect((1.25, 1.0, 0.55))


# MAIN FIGURE

def make_framework_figure(data: DemoData, out_path: str):
    fig = plt.figure(figsize=(14.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --------------------------------------------------------
    # TOP ROW
    # --------------------------------------------------------
    top_y = 0.56
    box_h_top = 0.30
    box_w_top = 0.18
    gap_top = 0.05
    x_top0 = 0.05

    top_x = [
        x_top0,
        x_top0 + (box_w_top + gap_top),
        x_top0 + 2 * (box_w_top + gap_top),
    ]

    panels = {
        "trades":   (top_x[0], top_y, box_w_top, box_h_top),
        "graph":    (top_x[1], top_y, box_w_top, box_h_top),
        "hetero":   (top_x[2], top_y, box_w_top, box_h_top),

        "manifold": (0.18, 0.12, 0.20, 0.28),
        "spectral": (0.44, 0.12, 0.22, 0.28),
        "tox":      (0.72, 0.12, 0.18, 0.28),
    }

    for key in panels:
        add_panel(ax, *panels[key])

    y_mid_top = top_y + box_h_top / 2
    add_arrow(ax, top_x[0] + box_w_top, y_mid_top, top_x[1], y_mid_top)
    add_arrow(ax, top_x[1] + box_w_top, y_mid_top, top_x[2], y_mid_top)

    hetero_center_x = top_x[2] + box_w_top / 2
    add_arrow(ax, hetero_center_x, top_y, hetero_center_x, 0.40)

    manifold_mid_y = 0.12 + 0.28 / 2
    add_arrow(ax, 0.18 + 0.20, manifold_mid_y, 0.44, manifold_mid_y)

    spectral_mid_y = 0.12 + 0.28 / 2
    add_arrow(ax, 0.44 + 0.22, spectral_mid_y, 0.72, spectral_mid_y)

    # TOP ROW PANELS

    x, y, w, hh = panels["trades"]
    panel_title(ax, x, y, w, hh, "Trades")

    ax1 = fig.add_axes([x + 0.025, y + 0.06, w - 0.05, hh - 0.12])
    ax1.scatter(data.trade_xy[:, 0], data.trade_xy[:, 1], s=20, color=COL_TRADE)
    clean_axis(ax1)
    ax1.set_aspect("equal")

    ax.text(x + w / 2, y + 0.03, "Independent trade observations",
            ha="center", fontsize=9.2)

    x, y, w, hh = panels["graph"]
    panel_title(ax, x, y, w, hh, "Trade similarity graph")

    ax2 = fig.add_axes([x + 0.025, y + 0.08, w - 0.05, hh - 0.15])
    draw_trade_similarity_graph(ax2, data.trade_xy, data.W)

    ax.text(
        x + w / 2,
        y + 0.03,
        r"$(\mathbf{W}_{TT})_{ij}=K_t(i,j)K_s(i,j)$",
        ha="center",
        fontsize=10.0
    )

    x, y, w, hh = panels["hetero"]
    panel_title(ax, x, y, w, hh, "Heterogeneous graph")

    ax5 = fig.add_axes([x + 0.025, y + 0.06, w - 0.05, hh - 0.12])
    draw_hetero_graph(ax5, n_clients_show=6, n_trades_show=18)

    ax.text(x + 0.03, y + 0.03, r"$\mathbf{A}=$", ha="left", va="center", fontsize=10.8)
    draw_block_matrix(ax, x + w * 0.62, y + 0.03, scale=0.85)

    # SECOND ROW

    x, y, w, hh = panels["manifold"]
    panel_title(ax, x, y, w, hh, "Graph manifold and Laplacian")

    draw_manifold_surface(fig, [x + 0.01, y + 0.05, w - 0.02, hh - 0.10])

    ax.text(
        x + w / 2,
        y + 0.03,
        r"$\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$",
        ha="center",
        fontsize=10.2,
    )

    x, y, w, hh = panels["spectral"]
    panel_title(ax, x, y, w, hh, "Spectral filtering")

    ax4 = fig.add_axes([x + 0.05, y + 0.11, w - 0.10, 0.09])

    idx = np.arange(45)
    raw = data.y_raw[:45]
    lp = data.y_lp[:45]
    hp = data.y_hp[:45]

    raw = raw / (np.max(np.abs(raw)) + 1e-12)
    lp = lp / (np.max(np.abs(lp)) + 1e-12)
    hp = hp / (np.max(np.abs(hp)) + 1e-12)

    off_raw = 1.8
    off_lp = 0.0
    off_hp = -1.8

    ax4.plot(idx, raw + off_raw, lw=1.4, color="#4C78A8")
    ax4.plot(idx, lp + off_lp, lw=2.0, color="#F58518")
    ax4.plot(idx, hp + off_hp, lw=1.3, color="#54A24B")

    ax4.text(-7.0, off_raw, r"$f$", fontsize=9.8, va="center", ha="left", clip_on=False)
    ax4.text(-7.0, off_lp, r"$f_{\mathrm{LP}}$", fontsize=9.8, va="center", ha="left", clip_on=False)
    ax4.text(-7.0, off_hp, r"$f_{\mathrm{HP}}$", fontsize=9.8, va="center", ha="left", clip_on=False)

    ax4.set_xlim(-1, 45)
    ax4.set_ylim(-2.4, 2.4)
    ax4.set_xticks([])
    ax4.set_yticks([])
    for s in ax4.spines.values():
        s.set_visible(False)

    ax.text(x + w / 2, y + 0.048, r"$f_{\mathrm{LP}}=e^{-t\mathbf{L}}f$", ha="center", fontsize=10.0)
    ax.text(x + w / 2, y + 0.020, r"$f_{\mathrm{HP}}=f-f_{\mathrm{LP}}$", ha="center", fontsize=10.0)

    x, y, w, hh = panels["tox"]
    panel_title(ax, x, y, w, hh, "Counterparty toxicity")

    ax6 = fig.add_axes([x + 0.006, y + 0.05, w - 0.012, hh - 0.10])
    draw_toxicity_vector_panel(
        ax6,
        data.theta_lp[:5],
        data.theta_hp[:5],
        data.theta_p[:5],
        client_names=[f"CP{i}" for i in range(5)],
    )

    ax.text(x + w / 2, y + 0.03, r"$T_c=(S_c,R_c,P_c)$",
            ha="center", fontsize=10.2)

    fig.savefig(out_path)
    plt.close(fig)


# MAIN

def main():
    data = make_demo_data()
    out = os.path.join(OUT_DIR, "fig_bipartite_spectral_graph_framework_pipeline.pdf")
    make_framework_figure(data, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()