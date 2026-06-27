"""
data_cleaning.py
-----------------
Step-by-step data cleaning pipeline for the comprehensive statistics dataset.
Each function corresponds to one cleaning decision made during exploratory
analysis (see notebooks/01_data_cleaning.ipynb for the full reasoning).
"""

import numpy as np
import pandas as pd
from scipy import stats


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw, uncleaned dataset."""
    return pd.read_csv(path)


def clean_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    income cleaning - exact steps followed during exploratory analysis:

    1. IQR on raw income flagged 78 outliers - too many, because income
       is naturally right-skewed (log-normal), so the raw IQR method
       over-flags normal high-income values as outliers.
    2. IQR on log-transformed income flagged a more reasonable 17 outliers.
       Inspecting those values showed a clear split: 4 were extreme,
       implausible values (3-5x the realistic maximum) - genuine errors -
       while the rest were just naturally high incomes.
    3. The actual correction applied: every value above the RAW upper
       IQR fence is capped (winsorized) at that fence. This caps the 4
       genuine errors AND a handful of naturally high (but legitimate)
       incomes - a simpler, slightly more conservative correction than
       only touching the 4 confirmed errors.
    4. Missing values are filled with the median (income is still skewed
       even after capping, so median is the more robust choice).
    """
    df = df.copy()

    # Step 1: IQR on raw income (for reference / to show why it over-flags)
    q1_raw, q3_raw = df['income'].quantile([0.25, 0.75])
    iqr_raw = q3_raw - q1_raw
    upper_bound_raw = q3_raw + 1.5 * iqr_raw
    lower_bound_raw = q1_raw - 1.5 * iqr_raw
    outliers_raw = df[(df['income'] < lower_bound_raw) | (df['income'] > upper_bound_raw)]
    # len(outliers_raw) was 78 - too many, income is naturally skewed

    # Step 2: IQR on log-transformed income (for inspection/diagnosis)
    log_income = np.log(df['income'].dropna())
    q1_log, q3_log = log_income.quantile([0.25, 0.75])
    iqr_log = q3_log - q1_log
    upper_bound_log = q3_log + 1.5 * iqr_log
    lower_bound_log = q1_log - 1.5 * iqr_log
    log_outlier_mask = (log_income < lower_bound_log) | (log_income > upper_bound_log)
    # len(log_income[log_outlier_mask]) was 17; top 4 were genuine errors,
    # the rest were naturally high incomes (see notebook for the inspection)

    # Step 3: winsorize - cap every value above the raw upper bound
    df.loc[df['income'] > upper_bound_raw, 'income'] = upper_bound_raw

    # Step 4: fill missing values with the median
    df['income'] = df['income'].fillna(df['income'].median())

    return df


def clean_height(df: pd.DataFrame) -> pd.DataFrame:
    """
    height_cm cleaning - exact steps followed:

    1. Shapiro-Wilk test confirmed height_cm is normally distributed
       (p-value > 0.05).
    2. Because the data is normal, outliers were checked with Z-score
       (not IQR) - 6 outliers found (values like 136-198 cm).
    3. Those 6 values were judged biologically plausible (real height
       range for humans), not data errors - so they were kept as-is.
    4. Missing values are filled with the mean, since the distribution
       is normal/symmetric.
    """
    df = df.copy()

    # Step 1: normality check (kept here for documentation/reproducibility)
    # stats.shapiro(df['height_cm'].dropna())  -> p-value ~0.79, normal

    # Step 2: Z-score outlier detection (since data is normal)
    height_no_na = df['height_cm'].dropna()
    z = pd.Series(stats.zscore(height_no_na), index=height_no_na.index)
    height_outliers = df.loc[z[abs(z) > 3].index]
    # len(height_outliers) was 6, values ranged ~136 to ~198 cm

    # Step 3: these are real, plausible heights - no correction applied

    # Step 4: fill missing values with the mean (normal distribution)
    df['height_cm'] = df['height_cm'].fillna(df['height_cm'].mean())

    return df


def clean_children_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    children_count cleaning - exact steps followed:

    1. Shapiro-Wilk test confirmed children_count is NOT normal
       (p-value extremely small) - expected, since it's a discrete
       Poisson-like count variable.
    2. Because the data is not normal, outliers were checked with IQR
       - 126 outliers found (values of 4, 5, or 6 children).
    3. Those values were judged real and plausible (people can have
       4-6 children), not data errors - so they were kept as-is.
    4. Missing values are filled with the median (non-normal data).
    """
    df = df.copy()

    # Step 1: normality check
    # stats.shapiro(df['children_count'].dropna())  -> p-value ~1.97e-35, not normal

    # Step 2: IQR outlier detection (since data is not normal)
    q1, q3 = df['children_count'].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    outliers = df[(df['children_count'] > upper_bound) | (df['children_count'] < lower_bound)]
    # len(outliers) was 126, all values were 4, 5, or 6

    # Step 3: these are real, plausible counts - no correction applied

    # Step 4: fill missing values with the median (non-normal distribution)
    df['children_count'] = df['children_count'].fillna(df['children_count'].median())

    return df


def fix_weight_and_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    weight_kg and bmi correction - this is a SEPARATE step, applied after
    the original generation script was found to have a bug.

    During descriptive statistics review, weight_kg showed an impossible
    distribution: ~82% of all values were clipped at the same minimum
    (45 kg), because the original formula (0.45*height - 40 + noise)
    produced unrealistic values for most people before clipping. This
    is a data-generation bug, not a real-world pattern, and cannot be
    meaningfully "cleaned" - it has to be regenerated with a realistic
    formula.

    Fix: weight is regenerated from a realistic baseline BMI (mean ~23,
    the normal/healthy range) combined with each person's actual height.
    bmi is then recalculated directly from the corrected height and
    weight, so by construction it is always 100% consistent (no
    separate consistency check against bmi is needed anymore).
    """
    df = df.copy()
    np.random.seed(42)
    n = len(df)

    target_bmi = np.random.normal(23, 4, n).clip(16, 40)
    weight = (target_bmi * (df['height_cm'] / 100) ** 2 + np.random.normal(0, 3, n))
    df['weight_kg'] = weight.clip(40, 130).round(1)
    df['bmi'] = (df['weight_kg'] / (df['height_cm'] / 100) ** 2).round(2)

    return df


def clean_job_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    job_satisfaction_1to5 is ordinal (Likert scale). Missing values are
    filled with the mode, since mean/median are not meaningful for
    ordinal data.
    """
    df = df.copy()
    mode_value = df['job_satisfaction_1to5'].mode()[0]
    df['job_satisfaction_1to5'] = df['job_satisfaction_1to5'].fillna(mode_value)
    return df


def clean_exam_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    exam_score must logically fall between 0 and 100. Values outside this
    range are impossible (data entry errors), so they are converted to NaN
    and then imputed with the median.
    """
    df = df.copy()
    invalid = (df['exam_score'] < 0) | (df['exam_score'] > 100)
    df.loc[invalid, 'exam_score'] = np.nan
    df['exam_score'] = df['exam_score'].fillna(df['exam_score'].median())
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert register_date to datetime and round-trip integer columns."""
    df = df.copy()
    df['register_date'] = pd.to_datetime(df['register_date'])
    df['job_satisfaction_1to5'] = df['job_satisfaction_1to5'].astype(int)
    df['children_count'] = df['children_count'].astype(int)
    return df


def check_duplicates(df: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    return df.duplicated().sum()


def full_cleaning_pipeline(path: str) -> pd.DataFrame:
    """Run the entire cleaning pipeline in the order it was developed."""
    df = load_raw_data(path)
    df = clean_income(df)
    df = clean_height(df)
    df = clean_children_count(df)
    df = clean_job_satisfaction(df)
    df = clean_exam_score(df)
    df = fix_weight_and_bmi(df)
    df = fix_data_types(df)
    return df


if __name__ == "__main__":
    df = full_cleaning_pipeline("../data/raw_dataset.csv")
    print(df.isna().sum())
    print("Duplicates:", check_duplicates(df))
