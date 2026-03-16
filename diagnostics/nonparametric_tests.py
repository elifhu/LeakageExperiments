#nonparametric_tests.py
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def mann_whitney_test(a, b):
    """
    Two-sided Mann–Whitney U test (Wilcoxon rank-sum).
    Returns dict with U, p, and sample sizes.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return {"U": float("nan"), "p": float("nan"), "n_a": len(a), "n_b": len(b)}

    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(U), "p": float(p), "n_a": len(a), "n_b": len(b)}