# ============================================================
# metrics.py
# Computes evaluation metrics and saves them to disk.
#
# Provides:
#   - compute_metrics: accuracy, precision, recall, F1 (binary + macro)
#   - save_metrics: dump metrics to JSON for the results folder
#   - plot_confusion_matrix: save a labeled confusion matrix as PNG
#   - print_metrics: nicely formatted console output
# ============================================================

import os
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display on EC2)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred) -> dict:
    # Cast to numpy arrays so this works whether inputs are lists, pandas, or torch tensors
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Per-class precision/recall/F1 (returns arrays of length 2 for binary)
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    # Macro-averaged (treats both classes equally regardless of support)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted average (weighted by class support — useful when imbalanced)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_class_0": float(p_per_class[0]),
        "recall_class_0": float(r_per_class[0]),
        "f1_class_0": float(f1_per_class[0]),
        "precision_class_1": float(p_per_class[1]),
        "recall_class_1": float(r_per_class[1]),
        "f1_class_1": float(f1_per_class[1]),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def save_metrics(metrics: dict, save_dir: str, name: str) -> str:
    # Save metrics to a JSON file with timestamp for tracking over runs
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"{name}_{timestamp}.json")

    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[metrics] Saved metrics to {file_path}")
    return file_path


def plot_confusion_matrix(y_true, y_pred, save_dir: str, name: str,
                          class_names: list = None) -> str:
    # Default class names for sarcasm task
    if class_names is None:
        class_names = ["Not Sarcasm", "Sarcasm"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Build the figure
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Heatmap with annotations
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {name}")

    # Save and close
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"confusion_{name}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=120)
    plt.close()

    print(f"[metrics] Saved confusion matrix to {file_path}")
    return file_path


def print_metrics(metrics: dict, name: str = ""):
    # Print a clean summary to console
    header = f" {name} ".center(60, "=") if name else "=" * 60
    print()
    print(header)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"F1 (macro)         : {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted)      : {metrics['f1_weighted']:.4f}")
    print(f"Precision (macro)  : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)     : {metrics['recall_macro']:.4f}")
    print(f"--- Per-class breakdown ---")
    print(f"Class 0 (Not Sarcasm): P={metrics['precision_class_0']:.4f} | R={metrics['recall_class_0']:.4f} | F1={metrics['f1_class_0']:.4f}")
    print(f"Class 1 (Sarcasm)    : P={metrics['precision_class_1']:.4f} | R={metrics['recall_class_1']:.4f} | F1={metrics['f1_class_1']:.4f}")
    print(f"Confusion matrix     : {metrics['confusion_matrix']}")
    print("=" * 60)


def get_classification_report(y_true, y_pred, target_names: list = None) -> str:
    # Sklearn's full classification report (useful for the final report write-up)
    if target_names is None:
        target_names = ["Not Sarcasm", "Sarcasm"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
