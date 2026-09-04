"""
data_loader.py
--------------
Load, clean, and preprocess the comprehensive statistics dataset.
Handles missing values and flags outliers using IQR and Z-score methods.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["register_date"] = pd.to_datetime(df["register_date"])
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of missing values per column."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for skewed numeric, mode for ordinal/categorical."""
    df = df.copy()
    df["income"] = df["income"].fillna(df["income"].median())
    df["height_cm"] = df["height_cm"].fillna(df["height_cm"].mean())
    df["job_satisfaction_1to5"] = df["job_satisfaction_1to5"].fillna(
        df["job_satisfaction_1to5"].mode()[0]
    )
    df["children_count"] = df["children_count"].fillna(df["children_count"].median())
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detect outliers in a numeric column using the IQR method."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers in a numeric column using the Z-score method."""
    z = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z) > threshold]


def cap_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cap outliers at the IQR fences (winsorizing) instead of dropping rows."""
    df = df.copy()
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[column] = df[column].clip(lower, upper)
    return df


def get_clean_dataset(path: str) -> pd.DataFrame:
    """Full pipeline: load -> impute -> cap key outliers."""
    df = load_data(path)
    df = impute_missing(df)
    df = cap_outliers(df, "income")
    df = cap_outliers(df, "exam_score")
    return df


if __name__ == "__main__":
    df = load_data("../data/comprehensive_dataset.csv")
    print(missing_value_report(df))
