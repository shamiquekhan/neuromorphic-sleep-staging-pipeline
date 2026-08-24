"""Evaluation metrics and visualization for sleep-stage classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    confusion_matrix, classification_report,
)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_all_metrics(y_true, y_pred, n_classes: int = 5):
    """Compute the full metric suite for sleep-stage evaluation."""
    per_cls_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    per_cls_recalls = []
    for c in range(n_classes):
        rec = f1_score(y_true, y_pred, labels=[c], average=None, zero_division=0)
        per_cls_recalls.append(float(rec[0]))
    recalls = np.clip(per_cls_recalls, 1e-12, 1.0)
    mgm = float(np.prod(recalls) ** (1.0 / n_classes))

    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "macro_gmean": round(mgm, 4),
        **{f"f1_{name.lower()}": round(float(per_cls_f1[i]), 4) for i, name in enumerate(STAGE_NAMES)},
    }


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot a normalized confusion matrix for exhibition."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    normalized = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, vmin=0, vmax=1)
    ax.set_xticks(range(5), STAGE_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), STAGE_NAMES)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title("Final Improved Student — normalized confusion matrix")

    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{normalized[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Row-normalized fraction")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def plot_per_class_f1(metrics: dict, save_path=None):
    """Bar chart of per-class F1 scores."""
    f1_values = [metrics.get(f"f1_{name.lower()}", 0) for name in STAGE_NAMES]
    df = pd.DataFrame({"Sleep stage": STAGE_NAMES, "F1": f1_values})

    ax = df.plot.bar(x="Sleep stage", y="F1", legend=False, figsize=(8, 4))
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Final Improved Student — per-class F1")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def top_confusions(cm, k: int = 5):
    """Rank the most common misclassifications."""
    rows = []
    for i in range(5):
        for j in range(5):
            if i != j and cm[i, j] > 0:
                rows.append({"true": STAGE_NAMES[i], "predicted": STAGE_NAMES[j], "count": int(cm[i, j])})
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)
