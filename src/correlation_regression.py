"""
correlation_regression.py
--------------------------
Correlation analysis (Pearson/Spearman) and regression models
(linear and logistic) using statsmodels for full inferential output.
"""

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def pearson_correlation(x, y, label: str = "") -> dict:
    """Pearson correlation coefficient (linear, parametric)."""
    r, p = stats.pearsonr(x, y)
    return {"pair": label, "method": "Pearson", "r": round(r, 4), "p_value": round(p, 4)}


def spearman_correlation(x, y, label: str = "") -> dict:
    """Spearman rank correlation (monotonic, non-parametric)."""
    r, p = stats.spearmanr(x, y)
    return {"pair": label, "method": "Spearman", "r": round(r, 4), "p_value": round(p, 4)}


def correlation_matrix(df: pd.DataFrame, columns: list, method: str = "pearson") -> pd.DataFrame:
    """Return a correlation matrix for the given numeric columns."""
    return df[columns].corr(method=method).round(3)


def linear_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """Fit an OLS linear regression model and return the fitted statsmodels result."""
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    return model


def logistic_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """Fit a logistic regression model and return the fitted statsmodels result."""
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.Logit(y, X).fit(disp=0)
    return model


def regression_diagnostics(model) -> dict:
    """Return key diagnostic metrics from a fitted OLS model."""
    return {
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_statistic": round(model.fvalue, 4),
        "f_pvalue": round(model.f_pvalue, 6),
        "aic": round(model.aic, 2),
        "bic": round(model.bic, 2),
    }
