<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:101B2D,100:173B4A&height=190&section=header&text=STATISTICAL%20ANALYSIS&fontSize=36&fontColor=F2F6FC&fontAlignY=42&desc=Python%20%7C%20Statistical%20Inference%20%7C%20Regression&descSize=15&descAlignY=68&descColor=62D6C5" width="100%" alt="Statistical Analysis — Python, Statistical Inference, Regression" />

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-2D8CFF?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Analysis-Statistical-0D9488?style=flat-square" alt="Statistical analysis" />
  <img src="https://img.shields.io/badge/License-MIT-64748B?style=flat-square" alt="MIT License" />
</p>

**A practical exploration of statistical methods using Python.**

</div>

---

This project explores a synthetic dataset of 2,000 observations and 22 variables using Python. It covers the main stages of a statistical analysis workflow, from data cleaning and descriptive statistics to hypothesis testing, correlation, and regression.

The analysis is organized into reusable Python modules, with a Jupyter notebook bringing the methods and results together. The dataset includes missing values, outliers, and several types of variables, making it useful for exploring how different statistical methods are applied and interpreted.

> **Dataset note**
>
> All data in this project is synthetic. The findings illustrate statistical methods on simulated data and should not be interpreted as evidence about real populations or medical outcomes.

## 01 · Project at a glance

<table>
  <tr>
    <td align="center" width="25%">
      <h2>2,000</h2>
      <sub>OBSERVATIONS</sub>
    </td>
    <td align="center" width="25%">
      <h2>22</h2>
      <sub>VARIABLES</sub>
    </td>
    <td align="center" width="25%">
      <h2>Python</h2>
      <sub>CORE LANGUAGE</sub>
    </td>
    <td align="center" width="25%">
      <h2>Modular</h2>
      <sub>CODE + NOTEBOOK</sub>
    </td>
  </tr>
</table>

## 02 · Selected findings

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

## 03 · Visual results

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Correlation structure</strong><br/>
      <sub>Relationships between numerical variables</sub><br/><br/>
      <img src="outputs/figures/heatmap_correlation.png" alt="Correlation heatmap of numerical variables" width="100%" />
    </td>
    <td align="center" width="50%">
      <strong>Group comparison</strong><br/>
      <sub>Distribution of outcomes across groups</sub><br/><br/>
      <img src="outputs/figures/box_anova_group.png" alt="Box plot comparing outcomes across groups" width="100%" />
    </td>
  </tr>
</table>

## 04 · Dataset

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

## 05 · Methods

<details>
<summary><strong>Data preparation and descriptive statistics</strong></summary>

- Missing-value reporting and imputation
- IQR-based outlier detection and outlier capping
- Mean, median, mode, variance, skewness, and kurtosis
- Frequency tables and confidence intervals
- Shapiro–Wilk normality tests, with attention to their sensitivity at large sample sizes

</details>

<details>
<summary><strong>Hypothesis testing and effect sizes</strong></summary>

- One-sample, independent-samples, and paired t-tests
- One-way ANOVA and Tukey HSD post-hoc comparisons
- Chi-square tests of independence
- Mann–Whitney U, Wilcoxon signed-rank, and Kruskal–Wallis tests
- Cohen's *d* and eta-squared

</details>

<details>
<summary><strong>Correlation, regression, and visualization</strong></summary>

- Pearson and Spearman correlation coefficients
- Simple and multiple ordinary least squares (OLS)
- Regression diagnostics and logistic regression with odds ratios
- Distribution plots, group comparisons, scatter plots, and correlation heatmaps

</details>

## 06 · Repository map

<details>
<summary><strong>View project files</strong></summary>

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

</details>

## 07 · Getting started

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/ParnazAli/statistical-analysis-project.git
cd statistical-analysis-project
pip install -r requirements.txt
```

**2. Launch the notebook**

Run these commands from the repository root:

```bash
cd notebooks
jupyter notebook statistical_analysis.ipynb
```

**3. Run the tests**

Open a separate terminal at the repository root:

```bash
pytest tests/ -v
```

## 08 · Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-2D8CFF?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/statsmodels-0D9488?style=flat-square" alt="statsmodels" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" alt="Seaborn" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" alt="pytest" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

---

<div align="center">

**Parnaz Ali**

[Email](mailto:parnazali1383@gmail.com) · [Repository](https://github.com/ParnazAli/statistical-analysis-project)

<sub>Released under the MIT License · See <a href="LICENSE">LICENSE</a> for details.</sub>

</div>
