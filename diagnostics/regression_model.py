#regression_model.py
from __future__ import annotations

import statsmodels.api as sm


def fit_ols(X, Y, robust=True):
    """
    OLS with optional HC3 robust covariance.
    X can be a Series/DataFrame/ndarray; constant is added automatically.
    """
    Xc = sm.add_constant(X)
    model = sm.OLS(Y, Xc)
    res = model.fit(cov_type="HC3" if robust else "nonrobust")
    return res