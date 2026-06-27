"""
hypothesis_tests.py
--------------------
Reusable hypothesis test functions, plus a documented record (in the
docstring of each section) of exactly which research question, which
columns, and what result each test produced in this project.

Research questions answered with this module:

1. Does the drug improve scores compared to placebo?
   columns: treatment (Drug/Placebo), score_after
   -> independent_ttest + mann_whitney + cohens_d_independent

2. Did scores actually change from before to after treatment?
   columns: score_before, score_after (same person, two time points)
   -> paired_ttest + wilcoxon_test + cohens_d_paired

3. Does performance differ across the three experimental groups?
   columns: group (A/B/C), performance
   -> one_way_anova + tukey_posthoc + eta_squared

4. Is smoking associated with having the disease?
   columns: smoker (Yes/No), has_disease (Yes/No)
   -> chi_square_test + cramers_v

5. Is there a relationship between weekly study hours and exam score?
   columns: study_hours_week, exam_score
   -> pearson_correlation + spearman_correlation
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------- Normality checks (used before every test below) ----------

def test_normality(x: pd.Series) -> dict:
    """
    Shapiro-Wilk normality test.

    Used on: score_after (split by treatment), performance (split by group),
    study_hours_week, exam_score.

    Results found in this project:
      - score_after | treatment == 'Drug'    -> p = 0.0146 (borderline, not normal)
      - score_after | treatment == 'Placebo' -> p = 0.3069 (normal)
      - performance | group A, B, C          -> all normal (p = 0.91, 0.07, 0.73)
      - study_hours_week                     -> p ~ 9.19e-36 (not normal, skewed)
      - exam_score                           -> p ~ 8.42e-20 (not normal)
    """
    stat, p = stats.shapiro(x.dropna())
    return {"statistic": stat, "p_value": p, "is_normal (alpha=0.05)": p > 0.05}


# ---------- Question 1: drug vs placebo on score_after ----------
# df['treatment'] == 'Drug' / 'Placebo', column: score_after
# Result: t = 12.366, p = 6.75e-34 (Welch) | U test p = 1.83e-32 | d = 0.553

def independent_ttest(group1, group2) -> dict:
    """
    Welch's independent samples t-test (does not assume equal variances).

    drug_group = df[df['treatment'] == 'Drug']['score_after']
    placebo_group = df[df['treatment'] == 'Placebo']['score_after']
    independent_ttest(drug_group, placebo_group)
    -> statistic=12.366, p_value=6.75e-34 (mean 68.25 vs 61.05)
    """
    stat, p = stats.ttest_ind(group1, group2, equal_var=False)
    return {"test": "Independent t-test (Welch)", "statistic": stat, "p_value": p}


def mann_whitney(group1, group2) -> dict:
    """
    Mann-Whitney U test: non-parametric alternative to independent t-test.
    Used because score_after was borderline non-normal in the Drug group.

    mann_whitney(drug_group, placebo_group) -> p_value=1.83e-32 (agrees with t-test)
    """
    stat, p = stats.mannwhitneyu(group1, group2)
    return {"test": "Mann-Whitney U", "statistic": stat, "p_value": p}


def cohens_d_independent(group1, group2) -> float:
    """
    Cohen's d for two independent groups (pooled standard deviation).

    cohens_d_independent(drug_group, placebo_group) -> 0.5526 (medium effect)
    """
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(), group2.std()
    pooled_std = (((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) ** 0.5
    return (group1.mean() - group2.mean()) / pooled_std


# ---------- Question 2: score_before vs score_after (paired) ----------
# columns: df['score_before'], df['score_after'] (same individuals)
# Result: t = -34.62, p = 3.48e-206 | d = 0.774 (manual formula)

def paired_ttest(before, after) -> dict:
    """
    Paired samples t-test.

    paired_ttest(df['score_before'], df['score_after'])
    -> statistic=-34.622, p_value=3.48e-206 (scores rose from before to after)
    """
    stat, p = stats.ttest_rel(before, after)
    return {"test": "Paired t-test", "statistic": stat, "p_value": p}


def wilcoxon_test(before, after) -> dict:
    """
    Wilcoxon signed-rank test: non-parametric alternative to paired t-test.

    wilcoxon_test(df['score_before'], df['score_after'])
    """
    stat, p = stats.wilcoxon(before, after)
    return {"test": "Wilcoxon signed-rank", "statistic": stat, "p_value": p}


def cohens_d_paired(before, after) -> float:
    """
    Cohen's d for paired samples (mean of differences / std of differences).
    Note: pingouin's pg.compute_effsize(..., paired=True) gave a different
    value (-0.372) for the same data, because it uses a different pooled
    standard deviation in its denominator. Both are reported in the report
    for transparency - there is no single universal formula for paired d.

    diff = df['score_after'] - df['score_before']
    cohens_d_paired(df['score_before'], df['score_after']) -> 0.7742
    """
    diff = after - before
    return diff.mean() / diff.std()


# ---------- Question 3: performance across group A/B/C ----------
# columns: df['group'] (A/B/C), df['performance']
# Result: F = 105.52, p = 2.73e-44, eta^2 = 0.0956
# Tukey: A-B, A-C, B-C all significant (B highest, C lowest)

def one_way_anova(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """
    One-way ANOVA across groups.

    one_way_anova(df, value_col='performance', group_col='group')
    -> statistic=105.52, p_value=2.73e-44
    """
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    stat, p = stats.f_oneway(*groups)
    return {"test": "One-way ANOVA", "statistic": stat, "p_value": p}


def kruskal_wallis(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Kruskal-Wallis H-test: non-parametric alternative to one-way ANOVA."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    stat, p = stats.kruskal(*groups)
    return {"test": "Kruskal-Wallis H", "statistic": stat, "p_value": p}


def tukey_posthoc(df: pd.DataFrame, value_col: str, group_col: str):
    """
    Tukey HSD post-hoc test after a significant ANOVA - identifies which
    specific pairs of groups differ.

    tukey_posthoc(df, value_col='performance', group_col='group')
    -> A-B: meandiff=4.88, reject=True
       A-C: meandiff=-3.02, reject=True
       B-C: meandiff=-7.90, reject=True  (B highest, C lowest)
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    return pairwise_tukeyhsd(df[value_col], df[group_col])


def eta_squared(df: pd.DataFrame, dv: str, between: str):
    """
    Eta squared effect size for one-way ANOVA, via pingouin.

    eta_squared(df, dv='performance', between='group')
    -> np2 (eta squared) = 0.0956 (medium-to-large effect)
    """
    import pingouin as pg
    return pg.anova(data=df, dv=dv, between=between, detailed=True)


# ---------- Question 4: smoker vs has_disease ----------
# columns: df['smoker'] (Yes/No), df['has_disease'] (Yes/No)
# Result: chi2 = 180.34, p = 4.09e-41, Cramer's V = 0.300

def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Chi-square test of independence between two categorical variables.

    chi_square_test(df, col1='smoker', col2='has_disease')
    -> statistic=180.336, p_value=4.09e-41
       (40.8% of smokers had the disease vs 12.5% of non-smokers)
    """
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return {"test": "Chi-square", "statistic": chi2, "p_value": p, "dof": dof, "expected": expected}


def cramers_v(chi2: float, n: int, table_shape: tuple) -> float:
    """
    Cramer's V effect size for the chi-square test.

    cramers_v(chi2=180.336, n=2000, table_shape=(2, 2)) -> 0.300 (medium effect)
    """
    r, c = table_shape
    return (chi2 / (n * min(r - 1, c - 1))) ** 0.5


# ---------- Question 5: study_hours_week vs exam_score ----------
# columns: df['study_hours_week'], df['exam_score']
# Result: Pearson r=0.829, Spearman rho=0.783, both p~0.0 (very strong, positive)

def pearson_correlation(x, y) -> dict:
    """
    Pearson correlation (linear, parametric).

    pearson_correlation(df['study_hours_week'], df['exam_score'])
    -> r=0.8287, p_value~0.0
    """
    r, p = stats.pearsonr(x, y)
    return {"method": "Pearson", "r": r, "p_value": p}


def spearman_correlation(x, y) -> dict:
    """
    Spearman rank correlation (monotonic, non-parametric). Used as the
    primary result here since study_hours_week is non-normal/skewed.

    spearman_correlation(df['study_hours_week'], df['exam_score'])
    -> rho=0.7830, p_value~0.0
    """
    rho, p = stats.spearmanr(x, y)
    return {"method": "Spearman", "rho": rho, "p_value": p}
