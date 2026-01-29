import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# Global settings
# --------------------------------------------------
warnings.filterwarnings("ignore")
REPORT_DIR = "reports"



def get_figure_dir(stage):
    path = os.path.join("reports", "figures", stage)
    os.makedirs(path, exist_ok=True)
    return path



# ==================================================
# MAIN ENTRY POINT (THIS FIXES YOUR ERROR)
# ==================================================
def evaluate_and_report(model, X, y, feature_names=None, threshold=0.317):
    print("\nMODEL EVALUATION METRICS")
    print("--------------------------------")

    proba = model.predict_proba(X)

    # ✅ FIX: handle 1D / 2D outputs
    if proba.ndim == 2:
        y_prob = proba[:, 1]
    else:
        y_prob = proba

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "AUC": roc_auc_score(y, y_prob),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1 Score": f1_score(y, y_pred)
    }

    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    print("--------------------------------")

    # save metrics
    import pandas as pd
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        "reports/training_evaluation_metrics.csv", index=False
    )

    # plots
    stage = "training"

    plot_precision_recall_safety(y, y_prob, stage)
    plot_threshold_optimization(y, y_prob, stage)
    plot_decision_buffer_zones(y_prob, stage)
    plot_confusion_matrix_cost(y, y_prob, stage)
    plot_performance_targets(metrics, stage)

    if feature_names is not None:
        base_model = model.get_feature_importance_model()

        plot_feature_importance(
            base_model,
            feature_names,
            stage
        )




    print("All training evaluation plots saved to reports/figures/training")
    print("Training Evaluation completed successfully\n")

    return metrics



# ==================================================
# 1. Safety-First Precision–Recall Trade-off
# ==================================================
def plot_precision_recall_safety(y_true, y_prob, stage, target_recall=0.91):
    fig_dir = get_figure_dir(stage)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, marker="o", linewidth=1.5)

    mask = (recall >= 0.90) & (recall <= 0.93)
    plt.scatter(recall[mask], precision[mask], color="orange", label="Safety Zone")

    plt.axvline(target_recall, linestyle="--", color="green", label="Target Recall (0.91)")
    plt.axhline(0.77, linestyle="--", color="blue", label="Target Precision (0.77)")

    if mask.any():
        idx = np.argmax(precision[mask])
        plt.scatter(recall[mask][idx], precision[mask][idx],
                    color="red", s=120, marker="*")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Safety-First: Precision–Recall Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/precision_recall_safety.png", dpi=300)
    plt.close()



# ==================================================
# 2. Threshold Optimization
# ==================================================
def plot_threshold_optimization(y_true, y_prob,stage):
    fig_dir = get_figure_dir(stage)

    thresholds = np.linspace(0.2, 0.6, 50)
    recalls, precisions = [], []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds)

        tp = cm[1, 1]
        fn = cm[1, 0]
        fp = cm[0, 1]

        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recalls.append(recall)
        precisions.append(precision)

    plt.figure(figsize=(7, 6))
    plt.plot(thresholds, recalls, label="Recall", color="green")
    plt.plot(thresholds, precisions, label="Precision", color="red")

    plt.axhline(0.90, linestyle="--", color="green", alpha=0.5)
    plt.axhline(0.93, linestyle="--", color="green", alpha=0.5)

    plt.axvline(0.317, color="black", linewidth=2, label="Optimal Threshold (0.317)")
    plt.axvspan(0.23, 0.35, color="orange", alpha=0.3, label="Safety Zone")

    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Optimization with Safety Constraints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/threshold_optimization.png", dpi=300)
    plt.close()


# ==================================================
# 3. Decision Buffer Zones
# ==================================================
def plot_decision_buffer_zones(y_prob,stage):
    fig_dir = get_figure_dir(stage)

    plt.figure(figsize=(7, 6))
    sns.kdeplot(y_prob, fill=True, color="blue")

    plt.axvspan(0.0, 0.2, color="green", alpha=0.2, label="Low Risk")
    plt.axvspan(0.2, 0.5, color="yellow", alpha=0.3, label="Moderate Risk")
    plt.axvspan(0.5, 1.0, color="red", alpha=0.2, label="High Risk")
    plt.axvline(0.317, linestyle="--", color="black", label="Binary Threshold")

    plt.xlabel("CAD Probability")
    plt.ylabel("Density")
    plt.title("Calibrated Decision Buffer Zones")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/decision_buffer_zones.png", dpi=300)
    plt.close()


# ==================================================
# 4. Feature Importance
# ==================================================
def plot_feature_importance(model, feature_names, stage, top_n=15):
    fig_dir = get_figure_dir(stage)

    importances = model.feature_importances_

    # 🔒 SAFETY CHECK (prevents crash forever)
    n = min(len(importances), len(feature_names), top_n)

    idx = np.argsort(importances)[::-1][:n]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1]
    )

    plt.xlabel("Importance")
    plt.title("Top Feature Importance (LightGBM)")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/feature_importance.png", dpi=300)
    plt.close()



# ==================================================
# 5. Confusion Matrix with Clinical Cost
# ==================================================
def plot_confusion_matrix_cost(y_true, y_prob,stage, threshold=0.317):
    fig_dir = get_figure_dir(stage)

    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)

    fn_cost = cm[1, 0] * 5
    fp_cost = cm[0, 1] * 1
    total_cost = fn_cost + fp_cost

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"Confusion Matrix (Clinical Cost = {total_cost})")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/confusion_matrix_cost.png", dpi=300)
    plt.close()


# ==================================================
# 6. Performance vs 91/77 Targets
# ==================================================
def plot_performance_targets(metrics, stage):
    fig_dir = get_figure_dir(stage)

    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color="green")
    plt.axhline(0.91, linestyle="--", color="gray", label="Recall Target")
    plt.axhline(0.77, linestyle="--", color="blue", label="Precision Target")

    plt.ylabel("Score")
    plt.title("Performance vs 91/77 Targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/performance_vs_targets.png", dpi=300)
    plt.close()
