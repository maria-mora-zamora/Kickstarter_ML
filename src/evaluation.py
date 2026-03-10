"""
src/evaluation.py
Unified evaluation function for all models in the Kickstarter ML project.
Prints metrics, saves plots, and appends results to a running CSV.
"""
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "results"
RESULTS_CSV = RESULTS_DIR / "all_model_results.csv"

SUCCESS_COLOR = "#3B82F6"
FAILURE_COLOR = "#EF4444"


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    figures_dir: str = None,
    results_csv: str = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a fitted classifier and save plots + metrics.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_test : array-like
    y_test : array-like
    model_name : str  — used in plot titles and CSV
    figures_dir : str — where to save plots (default: outputs/figures/)
    results_csv : str — path to append results (default: outputs/results/all_model_results.csv)
    threshold : float — decision threshold (default 0.5)

    Returns
    -------
    dict of all metric values
    """
    fdir = Path(figures_dir) if figures_dir else FIGURES_DIR
    rcsv = Path(results_csv) if results_csv else RESULTS_CSV
    fdir.mkdir(parents=True, exist_ok=True)
    rcsv.parent.mkdir(parents=True, exist_ok=True)

    # --- Predictions ---
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    # --- Core metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_bin = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception:
        pr_auc = float("nan")

    # --- Print report ---
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Model: {model_name}")
    print(sep)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 (binary)     : {f1_bin:.4f}")
    print(f"  F1 (macro)      : {f1_mac:.4f}")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  PR-AUC          : {pr_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted Fail", "Predicted Success"],
        yticklabels=["Actual Fail", "Actual Success"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    cm_path = fdir / f"cm_{safe_name}.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- ROC curve plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=SUCCESS_COLOR, lw=2,
                label=f"{model_name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center")
    plt.tight_layout()
    roc_path = fdir / f"roc_{safe_name}.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Precision-Recall curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec_arr, prec_arr, color=FAILURE_COLOR, lw=2,
                label=f"{model_name} (PR-AUC={pr_auc:.3f})")
        ax.axhline(y=y_test.mean(), color="k", linestyle="--",
                   label=f"Baseline (={y_test.mean():.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center")
    plt.tight_layout()
    pr_path = fdir / f"pr_{safe_name}.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Append to results CSV ---
    result_row = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_binary": round(f1_bin, 4),
        "f1_macro": round(f1_mac, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": threshold,
    }
    result_df = pd.DataFrame([result_row])
    if rcsv.exists():
        existing = pd.read_csv(rcsv)
        existing = existing[existing["model"] != model_name]
        result_df = pd.concat([existing, result_df], ignore_index=True)
    result_df.to_csv(rcsv, index=False)

    print(f"\n  Plots saved to: {fdir}")
    print(f"  Results appended to: {rcsv}")
    print(sep)

    return result_row
