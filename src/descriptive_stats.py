"""
descriptive_stats.py
---------------------
Functions for descriptive statistics: central tendency, dispersion,
shape (skewness/kurtosis), and frequency tables.
"""

import pandas as pd
from scipy import stats


def numeric_summary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Return mean, median, mode, std, variance, skewness, and kurtosis."""
    rows = []
    for col in columns:
        data = df[col].dropna()
        rows.append({
            "variable": col,
            "n": data.shape[0],
            "mean": data.mean(),
            "median": data.median(),
            "mode": data.mode().iloc[0] if not data.mode().empty else None,
            "std": data.std(),
            "variance": data.var(),
            "min": data.min(),
            "max": data.max(),
            "skewness": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
        })
    return pd.DataFrame(rows).round(3)


def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return frequency and relative frequency for a categorical column."""
    counts = df[column].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percentage": pct})


def confidence_interval_mean(data, confidence: float = 0.95):
    """Compute the confidence interval for the mean of a numeric sample."""
    data = pd.Series(data).dropna()
    n = len(data)
    mean = data.mean()
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin, mean + margin
