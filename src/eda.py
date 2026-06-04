"""
eda.py
------
Step 2 of the ML pipeline — Exploratory Data Analysis.

Generates and saves the following plots to outputs/plots/:
  1.  class_distribution.png      — bar chart of Stage counts
  2.  missing_values.png          — heatmap of missing data before cleaning
  3.  numerical_distributions.png — histograms of all numerical features
  4.  correlation_heatmap.png     — Pearson correlation matrix
  5.  boxplots_by_stage.png       — key features vs Stage
  6.  age_by_stage.png            — Age distribution per cirrhosis stage
  7.  pairplot_key_features.png   — pairplot of top clinical features

Run this BEFORE preprocessing (on raw data) to understand the original data,
or after preprocessing on the cleaned data for model-ready insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving to files
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_RAW_PATH, PLOTS_DIR, TARGET_COLUMN

# ── consistent style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_SM  = (8, 5)
FIGSIZE_MED = (14, 8)
FIGSIZE_LG  = (18, 12)


def _save(fig, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT]  Saved → {path}")


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart showing number of patients per Stage (1–4)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SM)
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    colors = sns.color_palette("Set2", len(counts))
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Class Distribution — Liver Cirrhosis Stages", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Cirrhosis Stage", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "01_class_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Heatmap showing which columns have missing values and how many."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("  [EDA]   No missing values found — skipping missing values plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.5)))
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"Count": missing, "Percent (%)": missing_pct})
    missing_df.sort_values("Percent (%)", ascending=True).plot(
        kind="barh", ax=ax, color=["#E07B7B", "#F5C26B"]
    )
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count / Percentage", fontsize=12)
    _save(fig, "02_missing_values.png")


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Histograms + KDE for all numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COLUMN]

    cols_per_row = 4
    n_rows = (len(num_cols) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 4, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#5B9BD5", edgecolor="white")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "03_numerical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap of all numerical features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(num_cols) * 0.9), max(8, len(num_cols) * 0.9)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"size": 8}, vmin=-1, vmax=1, center=0
    )
    ax.set_title("Pearson Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "04_correlation_heatmap.png")


def plot_boxplots_by_stage(df: pd.DataFrame) -> None:
    """
    Box plots of key clinical features grouped by cirrhosis Stage.
    Shows how each feature separates the classes.
    """
    key_features = ["Bilirubin", "Albumin", "Prothrombin", "Copper",
                    "SGOT", "Alk_Phos", "Platelets", "Cholesterol"]
    key_features = [f for f in key_features if f in df.columns]

    if not key_features:
        print("  [EDA]   Key features not found — skipping boxplots.")
        return

    cols_per_row = 4
    n_rows = (len(key_features) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 4.5, n_rows * 4))
    axes = axes.flatten()

    palette = sns.color_palette("Set2", 4)
    for i, feat in enumerate(key_features):
        ax = axes[i]
        sns.boxplot(x=TARGET_COLUMN, y=feat, data=df, palette=palette, ax=ax,
                    order=sorted(df[TARGET_COLUMN].unique()),
                    linewidth=1.2, flierprops={"marker": "o", "markersize": 4})
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("Stage", fontsize=9)
        ax.set_ylabel(feat, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Key Clinical Features by Cirrhosis Stage", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "05_boxplots_by_stage.png")


def plot_age_by_stage(df: pd.DataFrame) -> None:
    """Violin + strip plot of Age distribution across Stages."""
    age_col = "Age_Years" if "Age_Years" in df.columns else "Age"
    if age_col not in df.columns:
        print("  [EDA]   Age column not found — skipping age plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SM)
    palette = sns.color_palette("pastel", 4)
    sns.violinplot(x=TARGET_COLUMN, y=age_col, data=df,
                   order=sorted(df[TARGET_COLUMN].unique()),
                   palette=palette, inner="quartile", ax=ax, linewidth=1.2)
    sns.stripplot(x=TARGET_COLUMN, y=age_col, data=df,
                  order=sorted(df[TARGET_COLUMN].unique()),
                  color="black", alpha=0.3, size=3, ax=ax)
    ax.set_title("Age Distribution Across Cirrhosis Stages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cirrhosis Stage", fontsize=12)
    ax.set_ylabel("Age (Years)", fontsize=12)
    _save(fig, "06_age_by_stage.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot of top 5 clinical features coloured by Stage."""
    top_features = ["Bilirubin", "Albumin", "Prothrombin", "Copper", "SGOT"]
    top_features = [f for f in top_features if f in df.columns]

    if len(top_features) < 2:
        print("  [EDA]   Not enough features for pairplot — skipping.")
        return

    plot_df = df[top_features + [TARGET_COLUMN]].copy()
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].astype(str)

    g = sns.pairplot(plot_df, hue=TARGET_COLUMN,
                     palette="Set1", diag_kind="kde",
                     plot_kws={"alpha": 0.5, "s": 30})
    g.figure.suptitle("Pairplot — Key Clinical Features by Stage", y=1.01,
                       fontsize=14, fontweight="bold")
    _save(g.figure, "07_pairplot_key_features.png")


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the dataset to the console."""
    print("\n" + "─"*55)
    print("  DATASET SUMMARY")
    print("─"*55)
    print(f"  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"\n  Stage distribution:")
    for stage, count in df[TARGET_COLUMN].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    Stage {stage} → {count:>3} patients  ({pct:.1f}%)")
    print(f"\n  Numerical columns  ({len(df.select_dtypes(include=[np.number]).columns)}):")
    print(f"    {list(df.select_dtypes(include=[np.number]).columns)}")
    print(f"\n  Categorical columns ({len(df.select_dtypes(include=['object']).columns)}):")
    print(f"    {list(df.select_dtypes(include=['object']).columns)}")
    print("─"*55 + "\n")


def run_eda(df: pd.DataFrame = None) -> None:
    """
    Execute all EDA steps.
    Pass a DataFrame directly, or leave None to load from raw path.
    """
    print("\n" + "="*55)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*55)

    if df is None:
        df = pd.read_csv(DATASET_RAW_PATH)
        # drop NA stages for EDA on valid rows
        df = df[df[TARGET_COLUMN].astype(str).str.strip() != "NA"]
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        if "N_Days" in df.columns:
            df["Age_Years"] = (df["N_Days"] / 365.25).round(1)

    print_summary(df)

    print("  Generating plots...")
    plot_class_distribution(df)
    plot_missing_values(df)
    plot_numerical_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_stage(df)
    plot_age_by_stage(df)
    plot_pairplot(df)

    print(f"\n[DONE]  All EDA plots saved to → {PLOTS_DIR}")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_eda()
