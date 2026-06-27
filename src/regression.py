"""
regression.py
--------------
Linear and logistic regression functions, documented with the exact
research question, columns, and results found in this project.

Research questions answered with this module:

1. Does weekly study time predict exam score? (simple linear regression)
   columns: x = study_hours_week, y = exam_score
   -> linear_regression(df, 'exam_score', ['study_hours_week'])

2. Does age add predictive power beyond study hours? (multiple regression)
   columns: x = study_hours_week, age | y = exam_score
   -> linear_regression(df, 'exam_score', ['study_hours_week', 'age'])

3. Do exam score and study hours predict passing the exam? (logistic regression)
   columns: x = exam_score, study_hours_week | y = passed_exam (0/1)
   -> logistic_regression(df, 'passed_exam', ['exam_score', 'study_hours_week'])
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------- Question 1: simple linear regression ----------
# x = study_hours_week, y = exam_score
# Result: exam_score = 40.383 + 2.894 * study_hours_week
# R^2 = 0.6868, F = 4381, p < 0.001 (R^2 matches Pearson r^2 = 0.829^2)

def linear_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    Fit an OLS linear regression model. Works for both simple (one x
    column) and multiple (several x columns) regression.

    Simple regression used in this project:
      linear_regression(df, y_col='exam_score', x_cols=['study_hours_week'])
      -> const=40.383, study_hours_week=2.894, R^2=0.687
         (each extra hour of weekly study -> ~2.89 point increase in exam_score)

    Multiple regression used in this project:
      linear_regression(df, y_col='exam_score', x_cols=['study_hours_week', 'age'])
      -> const=39.051, study_hours_week=2.897 (p<0.001), age=0.038 (p=0.051, NOT significant)
         R^2 stayed 0.687 - adding age did not improve the model, consistent
         with age having no real effect on exam_score by design of the dataset.
    """
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    return sm.OLS(y, X).fit()


# ---------- Question 3: logistic regression ----------
# x = exam_score, study_hours_week | y = passed_exam (0/1)
# Result: exam_score coef=0.0843 (p<0.001, OR=1.088)
#         study_hours_week coef=-0.0196 (p=0.414, NOT significant - collinear with exam_score)
# Pseudo R^2 (McFadden) = 0.1625

def logistic_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """
    Fit a logistic regression model for a binary outcome.

    logistic_regression(df, y_col='passed_exam', x_cols=['exam_score', 'study_hours_week'])
    -> exam_score: coef=0.0843, p<0.001 (significant predictor of passing)
       study_hours_week: coef=-0.0196, p=0.414 (not significant once exam_score
       is in the model, due to multicollinearity - the two columns correlate
       at r=0.829, so study_hours_week's effect is already captured by exam_score)
    """
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    return sm.Logit(y, X).fit()


def odds_ratios(logit_model) -> pd.Series:
    """
    Convert logistic regression coefficients to odds ratios (exp of coef),
    which are easier to interpret than raw log-odds coefficients.

    odds_ratios(logit_model) on the model above
    -> exam_score: OR=1.088 (each +1 point on exam_score -> ~8.8% higher odds of passing)
       study_hours_week: OR=0.981 (not meaningful - coefficient was not significant)
    """
    return np.exp(logit_model.params)
