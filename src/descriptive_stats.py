"""
descriptive_stats.py
---------------------
Descriptive statistics for numeric and categorical variables.
"""

import pandas as pd


def numeric_summary(x: pd.Series) -> dict:
    """Return mean, median, mode, variance, std, min, max, skewness, kurtosis
    for a single numeric column."""
    return {
        "mean": x.mean(),
        "median": x.median(),
        "mode": x.mode()[0],
        "variance": x.var(),
        "std": x.std(),
        "max": x.max(),
        "min": x.min(),
        "skewness": x.skew(),
        "kurtosis": x.kurtosis(),
    }


def numeric_summary_table(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply numeric_summary to a list of columns and return a combined table."""
    results = {col: numeric_summary(df[col]) for col in columns}
    return pd.DataFrame(results)


def categorical_summary(x: pd.Series) -> pd.DataFrame:
    """Return frequency and percentage table for a categorical column."""
    counts = x.value_counts()
    percentages = x.value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "percentage": percentages})
