# Statistical Analysis Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This project explores a synthetic dataset of 2,000 observations and 22 variables using Python. It covers the main stages of a statistical analysis workflow, from data cleaning and descriptive statistics to hypothesis testing, correlation, and regression.

The analysis is organized into reusable Python modules, with a Jupyter notebook bringing the methods and results together. The dataset includes missing values, outliers, and several types of variables, making it useful for exploring how different statistical methods are applied and interpreted.

**All data in this project is synthetic.** The findings illustrate statistical methods on simulated data and should not be interpreted as evidence about real populations or medical outcomes.

## Selected findings

| Research question | Method | Result |
|---|---|---|
| Do post-treatment scores differ between the drug and placebo groups? | Welch's independent t-test | The drug group had higher scores in this dataset (*t* = 12.37, *p* < .001, Cohen's *d* = 0.55). |
| Do scores differ before and after treatment? | Paired t-test | Post-treatment scores were higher on average (*t* = −34.62, *p* < .001, *d* = 0.77). |
| Does performance vary across groups A, B, and C? | One-way ANOVA and Tukey HSD | The overall group difference was significant (*F* = 105.52, *p* < .001, η² = 0.096); all three pairwise comparisons were significant. |
| Are smoking status and disease status associated? | Chi-square test of independence | The variables were significantly associated (χ² = 180.34, *p* < .001). |
| How is weekly study time related to exam scores? | Simple linear regression | Each additional weekly study hour was associated with an estimated 2.81-point increase in exam score (*R*² = 0.671). |
| Which variables predict passing the exam? | Logistic regression | Exam score was a significant predictor (odds ratio = 1.08 per point), while weekly study hours were not significant after exam score was included in the model. |

These results describe the simulated dataset, not causal effects in real-world settings. Statistical significance is reported alongside effect sizes where available, since a small *p*-value alone does not show the magnitude or practical importance of a difference.

The notebook also compares selected parametric tests with non-parametric alternatives. The results agree for the group comparisons examined, providing a useful cross-check rather than a guarantee that every model assumption is satisfied.

## Visual results

### Correlation between numerical variables

![Correlation heatmap](outputs/figures/heatmap_correlation.png)

### Performance across experimental groups

![Performance by group](outputs/figures/box_anova_group.png)

## Dataset

The dataset contains 2,000 rows and 22 columns, combining numerical and categorical variables.

| Variable type | Examples |
|---|---|
| Nominal | `gender`, `education`, `city`, `treatment`, `group` |
| Continuous | `age`, `height_cm`, `weight_kg`, `income`, `exam_score` |
| Ordinal | `job_satisfaction_1to5` |
| Count | `children_count` |
| Binary | `passed_exam`, `smoker`, `has_disease` |
| Date | `register_date` |

Missing values are included in selected columns, and outliers are present in variables such as income. The analysis examines these issues before proceeding to statistical testing. The main notebook loads `data/comprehensive_dataset.csv`.

## Methods

- **Data preparation:** missing-value reporting and imputation, IQR-based outlier detection, and outlier capping.
- **Descriptive analysis:** mean, median, mode, variance, skewness, kurtosis, frequency tables, and confidence intervals.
- **Distribution assessment:** Shapiro–Wilk normality tests, with attention to the sensitivity of normality tests at large sample sizes.
- **Hypothesis testing:** one-sample, independent-samples, and paired t-tests; one-way ANOVA; Tukey HSD post-hoc comparisons; and chi-square tests.
- **Non-parametric methods:** Mann–Whitney U, Wilcoxon signed-rank, and Kruskal–Wallis tests.
- **Effect sizes:** Cohen's *d* and eta-squared, alongside significance tests.
- **Correlation:** Pearson and Spearman coefficients and a correlation matrix.
- **Regression:** simple and multiple ordinary least squares (OLS), regression diagnostics, and logistic regression with odds ratios.
- **Visualization:** distributions, group comparisons, scatter plots, and correlation heatmaps.

## Project structure

```text
statistical-analysis-project/
├── data/
│   ├── cleaned_dataset.csv
│   ├── comprehensive_dataset.csv
│   ├── raw_dataset.csv
│   └── generate_raw_dataset.py
├── notebooks/
│   └── statistical_analysis.ipynb
├── outputs/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── correlation_regression.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── descriptive_stats.py
│   ├── hypothesis_tests.py
│   ├── regression.py
│   └── visualization.py
├── tests/
│   └── test_statistics.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

Clone the repository and install the required packages:

```bash
git clone https://github.com/ParnazAli/statistical-analysis-project.git
cd statistical-analysis-project
pip install -r requirements.txt
```

Run the notebook from its own directory so its relative data and source paths resolve correctly:

```bash
cd notebooks
jupyter notebook statistical_analysis.ipynb
```

To run the test suite, open a separate terminal at the repository root and use:

```bash
pytest tests/ -v
```

## Tech stack

Python · pandas · NumPy · SciPy · statsmodels · Matplotlib · Seaborn · pytest · Jupyter

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Parnaz Ali**

[Email](mailto:parnazali1383@gmail.com)
