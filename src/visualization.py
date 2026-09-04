"""
visualization.py
-----------------
Reusable plotting functions (matplotlib + seaborn) for the
statistical analysis notebook. Each function saves a figure to
outputs/figures/ and also returns the matplotlib Axes.
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
FIG_DIR = "../outputs/figures"


def plot_histogram(df, column, bins=30, save_as=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {column}")
    if save_as:
        fig.savefig(f"{FIG_DIR}/{save_as}", dpi=150, bbox_inches="tight")
    return ax


def plot_boxplot(df, value_col, group_col=None, save_as=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=group_col, y=value_col, ax=ax, palette="Set2")
    ax.set_title(f"{value_col} distribution" + (f" by {group_col}" if group_col else ""))
    if save_as:
        fig.savefig(f"{FIG_DIR}/{save_as}", dpi=150, bbox_inches="tight")
    return ax


def plot_scatter(df, x_col, y_col, hue=None, save_as=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, alpha=0.6)
    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter=False, color="red")
    ax.set_title(f"{y_col} vs {x_col}")
    if save_as:
        fig.savefig(f"{FIG_DIR}/{save_as}", dpi=150, bbox_inches="tight")
    return ax


def plot_correlation_heatmap(corr_matrix, save_as=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    if save_as:
        fig.savefig(f"{FIG_DIR}/{save_as}", dpi=150, bbox_inches="tight")
    return ax


def plot_bar_counts(df, column, save_as=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order, ax=ax, palette="muted")
    ax.set_title(f"Frequency of {column}")
    if save_as:
        fig.savefig(f"{FIG_DIR}/{save_as}", dpi=150, bbox_inches="tight")
    return ax
