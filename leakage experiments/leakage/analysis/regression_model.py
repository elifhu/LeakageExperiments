import statsmodels.api as sm


def fit_ols(X, Y, robust=True):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit(cov_type="HC3" if robust else "nonrobust")
    return results