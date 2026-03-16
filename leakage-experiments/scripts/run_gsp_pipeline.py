from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from config import GRAPH_CONFIG
from paths import TABLES_DIR, ensure_dirs
from leakage.dataset_builder import load_trade_dataset
from leakage.leakage_synth import (
    assign_synthetic_counterparties_continuous,
    make_leakage_target_f,
)
from leakage.leakage_graph import build_kernels, knn_mask, build_graph_w, local_energy
from leakage.bipartite import (
    build_trade_counterparty_incidence,
    build_hybrid_adjacency,
    normalized_laplacian_from_adjacency,
)
from leakage.diffusion import heat_diffusion_filter
from leakage.leakage_counterparty_scores import (
    counterparty_scores_lowpass,
    counterparty_scores_spiky_residual,
    counterparty_scores_from_trade_values,
    build_stacked_counterparty_trade_signal,
    build_counterparty_vector_table,
)
from leakage.leakage_local_energy import local_energy_from_adjacency


def save_latex_table(df, path, caption, label):
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, int, np.floating)):
            if not np.isfinite(v):
                return "--"
            a = abs(v)
            if a >= 1e4 or (a < 1e-3 and a > 0):
                return f"{v:.2e}"
            return f"{v:.3f}"
        return str(v)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + "c" * len(cols) + r"}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(v) for v in row.values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ensure_dirs()

    trades = load_trade_dataset("trade_table")
    trades = assign_synthetic_counterparties_continuous(trades)
    trades = make_leakage_target_f(trades)

    trades_g = trades.sort_values("time").head(GRAPH_CONFIG.nmax).reset_index(drop=True)

    f = trades_g["f"].to_numpy(dtype=float)
    f = f / max(np.quantile(np.abs(f), 0.10), 1e-12)

    k_time, k_state, dist2_state, tt, _ = build_kernels(
        trades_g,
        tau=GRAPH_CONFIG.tau,
        gamma=GRAPH_CONFIG.gamma,
    )

    dist2_time = ((tt[:, None] - tt[None, :]) / GRAPH_CONFIG.tau) ** 2
    dist2_state_scaled = dist2_state / GRAPH_CONFIG.gamma ** 2

    mask = knn_mask(dist2_time + dist2_state_scaled, k=GRAPH_CONFIG.k_nn, symmetric=True)

    w_tt = build_graph_w(k_time, k_state, mask).tocsr()
    w_tt = (w_tt + w_tt.T) * 0.5
    w_tt.setdiag(0)
    w_tt.eliminate_zeros()

    trade_e0, trade_e0_norm, _ = local_energy(w_tt, f)

    b, counterparty_id_to_row, _ = build_trade_counterparty_incidence(trades_g)
    n_c, _ = b.shape

    a = build_hybrid_adjacency(
        b,
        w_tt=w_tt,
        alpha_tt=GRAPH_CONFIG.alpha_tt,
        alpha_bt=GRAPH_CONFIG.alpha_bt,
    )
    laplacian = normalized_laplacian_from_adjacency(a)

    f_stack = build_stacked_counterparty_trade_signal(
        trades=trades_g,
        trade_signal=trade_e0_norm,
        counterparty_id_to_row=counterparty_id_to_row,
        counterparty_col="counterparty_id",
        how="mean",
        use_abs_counterparty=True,
    )

    rows = []
    counterparty_vector_target = None
    t_target = 120.0

    for t_diff in GRAPH_CONFIG.diffusion_times:
        f_hat, _ = heat_diffusion_filter(laplacian, f_stack, t=t_diff, k_eigs=GRAPH_CONFIG.k_eigs)

        f_hat_trade = f_hat[n_c:]

        e_hat, e_hat_norm, _ = local_energy_from_adjacency(a, f_hat)
        trade_e_hat = e_hat_norm[n_c:]

        scores_energy = counterparty_scores_from_trade_values(
            trades_g,
            trade_e_hat,
            counterparty_col="counterparty_id",
            how="mean",
            use_abs=True,
        )

        theta_lp = counterparty_scores_lowpass(f_hat, n_counterparties=n_c)

        scores_hp = counterparty_scores_spiky_residual(
            trades_g,
            f,
            f_hat_trade,
            counterparty_col="counterparty_id",
            how="mean",
            use_abs=True,
        )

        hp_vals = np.array(list(scores_hp.values())) if isinstance(scores_hp, dict) else np.asarray(scores_hp)

        smooth_hat = f_hat.T @ (laplacian @ f_hat)
        rayleigh_hat = smooth_hat / (f_hat @ f_hat + 1e-12)

        rows.append({
            "t_diff": t_diff,
            "rayleigh": rayleigh_hat,
            "mean_S_lp": np.nanmean(theta_lp),
            "mean_R_hp": np.nanmean(hp_vals),
        })

        counterparty_vector_t = build_counterparty_vector_table(
            theta_lp=theta_lp,
            scores_hp=scores_hp,
            scores_energy=scores_energy,
            counterparty_id_to_row=counterparty_id_to_row,
            t_diff=t_diff,
        )

        counterparty_vector_t_path = TABLES_DIR / f"counterparty_vectors_t{int(t_diff)}.csv"
        counterparty_vector_t.to_csv(counterparty_vector_t_path, index=False)

        if np.isclose(t_diff, t_target):
            counterparty_vector_target = counterparty_vector_t.copy()

    df_summary = pd.DataFrame(rows)

    df_summary_tex = df_summary.copy()
    df_summary_tex.columns = [r"$t$", r"Rayleigh", r"Mean LP", r"Mean HP"]

    summary_csv_path = TABLES_DIR / "table_hetero_diffusion_summary.csv"
    summary_tex_path = TABLES_DIR / "table_hetero_diffusion_summary.tex"

    df_summary.to_csv(summary_csv_path, index=False)

    save_latex_table(
        df_summary_tex,
        summary_tex_path,
        caption="Spectral diffusion summary on the heterogeneous trade--counterparty graph.",
        label="tab:hetero_diffusion_summary",
    )

    if counterparty_vector_target is not None:
        counterparty_csv_path = TABLES_DIR / "table_counterparty_vector.csv"
        counterparty_tex_path = TABLES_DIR / "table_counterparty_vector.tex"

        counterparty_vector_target.to_csv(counterparty_csv_path, index=False)

        counterparty_tex = counterparty_vector_target[
            ["rank", "counterparty_id", "S_lp", "R_hp", "P_energy"]
        ].head(10)

        save_latex_table(
            counterparty_tex,
            counterparty_tex_path,
            caption=(
                "Counterparty vector representation "
                r"$z_c=(S_c,R_c,P_c)$ at diffusion scale "
                rf"$t={t_target:.0f}$, where $S_c$ denotes low-pass propagation, "
                r"$R_c$ the high-pass residual component, and $P_c$ the local-energy score."
            ),
            label="tab:counterparty_vector",
        )

    print("Saved:")
    print(summary_csv_path)
    print(summary_tex_path)

    if counterparty_vector_target is not None:
        print(TABLES_DIR / "table_counterparty_vector.csv")
        print(TABLES_DIR / "table_counterparty_vector.tex")


if __name__ == "__main__":
    main()