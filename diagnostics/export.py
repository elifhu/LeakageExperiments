# export.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_state_regression_table_tex(
    model_state,
    outpath_tex: str,
    caption: str,
    label: str,
    feature_names: list[str] | None = None,
):
    """
    Creates a booktabs-style LaTeX table for coefficients (excluding intercept),
    writes to outputs/tables/*.tex.
    """
    # params: [const, x1, x2, ...]
    k = len(model_state.params) - 1
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(k)]
    if len(feature_names) != k:
        feature_names = feature_names[:k] + [f"x{i+1}" for i in range(len(feature_names), k)]

    rows = []
    for name, b, se, t, p in zip(
        feature_names,
        model_state.params[1:],
        model_state.bse[1:],
        model_state.tvalues[1:],
        model_state.pvalues[1:],
    ):
        rows.append([name, float(b), float(se), float(t), float(p)])

    df = pd.DataFrame(rows, columns=["Variable", "Estimate", "Std. Err.", "t-stat", "p-value"])

    tex = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4g",
        caption=caption,
        label=label,
        column_format="lrrrr",
        bold_rows=False,
    )

    outpath = Path(outpath_tex)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(tex)