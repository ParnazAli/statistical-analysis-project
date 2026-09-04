"""
test_statistics.py
-------------------
Unit tests for the src/ statistics package. Run with:

    pytest tests/

These are lightweight sanity checks, not a full test suite — they verify
that each function returns the expected structure and sensible values on
small, hand-checkable inputs (not the full dataset).
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from descriptive_stats import numeric_summary, frequency_table, confidence_interval_mean
from hypothesis_tests import (
    independent_ttest, paired_ttest, one_way_anova, chi_square_test,
    cohens_d, eta_squared, interpret_cohens_d,
)
from correlation_regression import pearson_correlation, correlation_matrix, linear_regression
from data_loader import detect_outliers_iqr, cap_outliers, missing_value_report


# ---------- descriptive_stats ----------

def test_numeric_summary_basic():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = numeric_summary(df, ["x"])
    assert result.loc[0, "mean"] == 3
    assert result.loc[0, "n"] == 5


def test_confidence_interval_contains_mean():
    data = pd.Series([10, 12, 11, 13, 9, 14, 10])
    low, high = confidence_interval_mean(data)
    assert low < data.mean() < high


# ---------- hypothesis_tests ----------

def test_cohens_d_zero_when_means_equal():
    g1 = pd.Series([1, 2, 3, 4, 5])
    g2 = pd.Series([1, 2, 3, 4, 5])
    assert abs(cohens_d(g1, g2)) < 1e-9


def test_cohens_d_interpretation_labels():
    assert interpret_cohens_d(0.05) == "negligible"
    assert interpret_cohens_d(0.3) == "small"
    assert interpret_cohens_d(0.6) == "medium"
    assert interpret_cohens_d(1.2) == "large"


def test_independent_ttest_detects_clear_difference():
    g1 = pd.Series(np.random.normal(50, 5, 200))
    g2 = pd.Series(np.random.normal(60, 5, 200))
    result = independent_ttest(g1, g2, label="test")
    assert result["p_value"] < 0.05
    assert result["significant (alpha=0.05)"]


def test_paired_ttest_no_change_not_significant():
    np.random.seed(1)
    before = pd.Series(np.random.normal(50, 5, 100))
    after = before + np.random.normal(0, 0.01, 100)  # tiny random noise, no real shift
    result = paired_ttest(before, after, label="no-change")
    assert result["p_value"] > 0.05


def test_one_way_anova_and_eta_squared_range():
    df = pd.DataFrame({
        "value": list(np.random.normal(0, 1, 50)) + list(np.random.normal(5, 1, 50)),
        "group": ["A"] * 50 + ["B"] * 50,
    })
    result = one_way_anova(df, "value", "group")
    assert result["p_value"] < 0.05
    assert 0 <= result["eta_squared"] <= 1


def test_chi_square_independent_categories():
    np.random.seed(0)
    df = pd.DataFrame({
        "a": np.random.choice(["x", "y"], 500),
        "b": np.random.choice(["p", "q"], 500),
    })
    result = chi_square_test(df, "a", "b")
    assert "p_value" in result
    assert result["dof"] == 1


# ---------- correlation_regression ----------

def test_pearson_correlation_perfect_linear():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([2, 4, 6, 8, 10])
    result = pearson_correlation(x, y, label="perfect")
    assert result["r"] == 1.0


def test_linear_regression_recovers_known_slope():
    x = np.arange(100)
    y = 3 * x + 7
    df = pd.DataFrame({"x": x, "y": y})
    model = linear_regression(df, "y", ["x"])
    assert abs(model.params["x"] - 3) < 1e-6
    assert abs(model.params["const"] - 7) < 1e-6


# ---------- data_loader ----------

def test_detect_outliers_iqr_flags_extreme_value():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 12, 11, 200]})
    outliers = detect_outliers_iqr(df, "x")
    assert 200 in outliers["x"].values


def test_cap_outliers_removes_extremes():
    df = pd.DataFrame({"x": [10, 11, 12, 13, 12, 11, 200]})
    capped = cap_outliers(df, "x")
    assert capped["x"].max() < 200


def test_missing_value_report_only_lists_missing_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missing_value_report(df)
    assert "a" in report.index
    assert "b" not in report.index
