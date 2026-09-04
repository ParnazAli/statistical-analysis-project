"""
hypothesis_tests.py
--------------------
Implements normality checks plus parametric and non-parametric
hypothesis tests, each returning a clean dict of results.
"""

import numpy as np
import pandas as pd
from scipy import stats


def test_normality(data, name: str = "") -> dict:
    """Shapiro-Wilk test for normality (best for n < 5000)."""
    data = pd.Series(data).dropna()
    stat, p = stats.shapiro(data)
    return {
        "variable": name,
        "test": "Shapiro-Wilk",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "is_normal (alpha=0.05)": p > 0.05,
    }


def cohens_d(group1, group2) -> float:
    """Cohen's d effect size for two independent samples (pooled SD)."""
    g1, g2 = pd.Series(group1).dropna(), pd.Series(group2).dropna()
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Rule-of-thumb qualitative label for a Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def eta_squared(anova_stat: float, df_between: int, df_within: int) -> float:
    """Effect size (eta squared) for a one-way ANOVA from the F-statistic and df's."""
    return (anova_stat * df_between) / (anova_stat * df_between + df_within)


def independent_ttest(group1, group2, label: str = "") -> dict:
    """Independent samples t-test (Welch's, does not assume equal variance),
    reported alongside Cohen's d effect size."""
    g1, g2 = group1.dropna(), group2.dropna()
    stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    d = cohens_d(g1, g2)
    return {
        "comparison": label,
        "test": "Independent t-test (Welch)",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
        "cohens_d": round(d, 3),
        "effect_size": interpret_cohens_d(d),
    }


def paired_ttest(before, after, label: str = "") -> dict:
    """Paired samples t-test (e.g., before vs after measurements),
    reported alongside Cohen's d for paired samples."""
    diff = pd.Series(after).reset_index(drop=True) - pd.Series(before).reset_index(drop=True)
    stat, p = stats.ttest_rel(before, after)
    d = diff.mean() / diff.std(ddof=1)
    return {
        "comparison": label,
        "test": "Paired t-test",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
        "cohens_d": round(d, 3),
        "effect_size": interpret_cohens_d(d),
    }


def one_way_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """One-way ANOVA across groups in group_col for value_col,
    reported alongside eta-squared effect size."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    stat, p = stats.f_oneway(*groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    eta2 = eta_squared(stat, df_between, df_within)
    return {
        "comparison": f"{value_col} by {group_col}",
        "test": "One-way ANOVA",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
        "eta_squared": round(eta2, 4),
    }


def tukey_posthoc(df: pd.DataFrame, value_col: str, group_col: str):
    """Tukey HSD post-hoc test after a significant ANOVA."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    result = pairwise_tukeyhsd(df[value_col], df[group_col])
    return result


def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Chi-square test of independence between two categorical variables."""
    table = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = stats.chi2_contingency(table)
    return {
        "comparison": f"{col1} vs {col2}",
        "test": "Chi-square test of independence",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "dof": dof,
        "significant (alpha=0.05)": p < 0.05,
    }


def mann_whitney(group1, group2, label: str = "") -> dict:
    """Mann-Whitney U test: non-parametric alternative to independent t-test."""
    stat, p = stats.mannwhitneyu(group1.dropna(), group2.dropna())
    return {
        "comparison": label,
        "test": "Mann-Whitney U",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
    }


def wilcoxon_test(before, after, label: str = "") -> dict:
    """Wilcoxon signed-rank test: non-parametric alternative to paired t-test."""
    stat, p = stats.wilcoxon(before, after)
    return {
        "comparison": label,
        "test": "Wilcoxon signed-rank",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
    }


def kruskal_wallis(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Kruskal-Wallis H-test: non-parametric alternative to one-way ANOVA."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    stat, p = stats.kruskal(*groups)
    return {
        "comparison": f"{value_col} by {group_col}",
        "test": "Kruskal-Wallis H",
        "statistic": round(stat, 4),
        "p_value": round(p, 4),
        "significant (alpha=0.05)": p < 0.05,
    }
