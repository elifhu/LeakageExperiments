from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import GRAPH_CONFIG
from paths import FIGURES_DIR, ensure_dirs
from leakage.dataset_builder import load_trade_dataset
from leakage.leakage_synth import (
    assign_synthetic_counterparties_continuous,
    make_leakage_target_f,
)
from leakage.leakage_graph import (
    build_kernels,
    knn_mask,
    build_graph_w,
    normalized_laplacian,
    dirichlet_energy,
    local_energy,
    largest_connected_component,
    eigmodes,
)
from leakage.leakage_plots import (
    plot_geometry_with_nodecolor,
    plot_spectral_embedding,
    plot_zlocal_over_time_highlight,
    safe_corr,
)


def save_pdf(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")


def main():
    ensure_dirs()

    trades = load_trade_dataset("trade_table")
    trades = assign_synthetic_counterparties_continuous(trades)
    trades = make_leakage_target_f(trades)

    trades_g = trades.sort_values("time").head(GRAPH_CONFIG.nmax).reset_index(drop=True)

    f_raw = trades_g["f"].to_numpy(dtype=float)
    f_scale = float(np.nanquantile(np.abs(f_raw), 0.10))
    f_scale = max(f_scale, 1e-12)
    f = f_raw / f_scale
    trades_g["f"] = f

    k_time, k_state, dist2_state, tt, _ = build_kernels(
        trades_g,
        tau=GRAPH_CONFIG.tau,
        gamma=GRAPH_CONFIG.gamma,
    )

    dist2_time = ((tt[:, None] - tt[None, :]) / GRAPH_CONFIG.tau) ** 2
    dist2_state_scaled = dist2_state / (GRAPH_CONFIG.gamma ** 2)
    dist2_combo = dist2_time + dist2_state_scaled

    mask = knn_mask(dist2_combo, k=GRAPH_CONFIG.k_nn, symmetric=True)
    w = build_graph_w(k_time, k_state, mask).tocsr()
    w = (w + w.T) * 0.5
    w.setdiag(0.0)
    w.eliminate_zeros()

    laplacian = normalized_laplacian(w)
    quad = dirichlet_energy(laplacian, f)
    rayleigh = float(quad / (f @ f + 1e-12))

    print("f^T L f =", float(quad))
    print("Rayleigh(f) =", rayleigh)

    e_local, e_local_norm, d = local_energy(w, f)
    e_local = np.maximum(e_local, 0.0)
    e_local_norm = np.maximum(e_local_norm, 0.0)

    rng = np.random.default_rng(42)
    b_reps = 200
    e_null = np.zeros((b_reps, len(f)), dtype=float)

    for b in range(b_reps):
        f_sh = rng.permutation(f)
        wf_sh = w @ f_sh
        wf2_sh = w @ (f_sh ** 2)
        e_null[b] = (d * (f_sh ** 2) - 2.0 * f_sh * wf_sh + wf2_sh) / (d + 1e-12)

    z_local = (e_local_norm - e_null.mean(axis=0)) / (e_null.std(axis=0) + 1e-12)

    print("corr(z_local, f) =", safe_corr(z_local, f))

    fig = plot_zlocal_over_time_highlight(
        trades_g["time"].values,
        z_local,
        threshold=2.0,
        title="Structural anomaly score over trade time",
    )
    save_pdf(fig, f"{FIGURES_DIR}/fig_zlocal_over_time.pdf")
    plt.close(fig)

    w_geom, keep_geom = largest_connected_component(w)
    g_geom = nx.from_scipy_sparse_array(w_geom)
    pos_geom = nx.spring_layout(g_geom, seed=42, k=0.25)

    liq_geom = trades_g.loc[keep_geom, "liq"].to_numpy(dtype=float)
    liq_geom = (liq_geom - np.nanmin(liq_geom)) / (np.nanmax(liq_geom) - np.nanmin(liq_geom) + 1e-12)

    fig = plot_geometry_with_nodecolor(
        g_geom,
        pos_geom,
        liq_geom,
        title="Trade similarity graph",
        cbar_label="Liquidity proxy (scaled)",
    )
    save_pdf(fig, f"{FIGURES_DIR}/fig_trade_graph_geometry.pdf")
    plt.close(fig)

    evals, evecs = eigmodes(laplacian, k=4)
    x_emb = evecs[:, 1]
    y_emb = evecs[:, 2]

    fig = plot_spectral_embedding(
        x_emb,
        y_emb,
        z_local,
        title="Spectral embedding of trade similarity graph",
        xlabel="Graph Fourier coordinate 1",
        ylabel="Graph Fourier coordinate 2",
        cbar_label="Structural anomaly score",
    )
    save_pdf(fig, f"{FIGURES_DIR}/fig_trade_graph_embedding.pdf")
    plt.close(fig)

    print("Saved figures:")
    print(f"{FIGURES_DIR}/fig_zlocal_over_time.pdf")
    print(f"{FIGURES_DIR}/fig_trade_graph_geometry.pdf")
    print(f"{FIGURES_DIR}/fig_trade_graph_embedding.pdf")


if __name__ == "__main__":
    main()