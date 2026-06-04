"""
model_evaluation.py
-------------------
Step 5 of the ML pipeline.

Responsibilities:
  - Evaluate all trained models on the test set
  - Print classification reports
  - Save accuracy comparison table to outputs/reports/
  - Generate and save:
      08_accuracy_comparison.png   — bar chart of all model accuracies
      09_confusion_matrix_rf.png   — confusion matrix for Random Forest
      10_feature_importance.png    — top-20 feature importances (RF only)
      11_roc_curves.png            — ROC curves (one-vs-rest)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PLOTS_DIR, RESULTS_SAVE_PATH

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT]  Saved → {path}")


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every model and return a comparison DataFrame.
    """
    print("\n" + "="*55)
    print("  STEP 5 — MODEL EVALUATION")
    print("="*55)

    results = []
    for name, model in trained_models.items():
        y_pred   = model.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": round(acc * 100, 2)})
        print(f"\n  ── {name} ──")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred,
                                    target_names=["Stage 1","Stage 2","Stage 3","Stage 4"],
                                    zero_division=0))

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    print("\n  ── Final Accuracy Ranking ──")
    print(results_df.to_string(index=False))
    return results_df


def save_results_csv(results_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"\n[SAVE]  Results table saved → {RESULTS_SAVE_PATH}")


def plot_accuracy_comparison(results_df: pd.DataFrame) -> None:
    """Horizontal bar chart comparing all model accuracies."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("RdYlGn", len(results_df))[::-1]
    bars = ax.barh(results_df["Model"], results_df["Accuracy"],
                   color=palette, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontweight="bold", fontsize=11)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison — Liver Cirrhosis Stage Prediction",
                 fontsize=13, fontweight="bold", pad=12)
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, "08_accuracy_comparison.png")


def plot_confusion_matrix(model, X_test, y_test,
                          model_name: str = "Random Forest") -> None:
    """Confusion matrix heatmap for the given model."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    labels = [f"Stage {i}" for i in sorted(y_test.unique())]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 13, "weight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted Stage", fontsize=12)
    ax.set_ylabel("Actual Stage",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, f"09_confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance(model, feature_names: list,
                            top_n: int = 20) -> None:
    """
    Horizontal bar chart of the top N feature importances.
    Only works for tree-based models (Random Forest, Gradient Boosting).
    """
    if not hasattr(model, "feature_importances_"):
        print("  [SKIP]  Feature importance not available for this model.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("viridis", len(feat_df))[::-1]
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1],
            color=palette, edgecolor="white")
    ax.set_xlabel("Importance Score (Gini)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "10_feature_importance.png")


def plot_roc_curves(trained_models: dict, X_test, y_test) -> None:
    """
    One-vs-Rest ROC curves for all models.
    Uses macro-averaged AUC.
    """
    classes    = sorted(y_test.unique())
    y_bin      = label_binarize(y_test, classes=classes)

    fig, axes  = plt.subplots(2, 3, figsize=(16, 10))
    axes       = axes.flatten()
    colors     = sns.color_palette("Set1", len(classes))

    for ax_idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[ax_idx]
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            ax.text(0.5, 0.5, "ROC N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            ax.set_title(name)
            continue

        try:
            auc = roc_auc_score(y_bin, y_score, multi_class="ovr", average="macro")
            for i, cls in enumerate(classes):
                RocCurveDisplay.from_predictions(
                    y_bin[:, i], y_score[:, i],
                    name=f"Stage {cls}", ax=ax, color=colors[i], lw=1.5
                )
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_title(f"{name}\n(AUC macro = {auc:.3f})",
                         fontsize=10, fontweight="bold")
        except Exception as e:
            ax.set_title(f"{name}\n(ROC error)")

        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("ROC Curves — One-vs-Rest per Model",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "11_roc_curves.png")


def run_evaluation(trained_models: dict, X_test, y_test, feature_names: list) -> pd.DataFrame:
    """Full evaluation pipeline."""
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    save_results_csv(results_df)

    print("\n  Generating evaluation plots...")
    plot_accuracy_comparison(results_df)
    plot_confusion_matrix(trained_models["Random Forest"], X_test, y_test, "Random Forest")
    plot_feature_importance(trained_models["Random Forest"], feature_names)
    plot_roc_curves(trained_models, X_test, y_test)

    print(f"\n[DONE]  All evaluation plots saved to → {PLOTS_DIR}")
    print("="*55 + "\n")
    return results_df
