import os
import warnings
import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Plotting utilities
from src.evaluation import (
    plot_precision_recall_safety,
    plot_threshold_optimization,
    plot_decision_buffer_zones,
    plot_confusion_matrix_cost,
    plot_performance_targets,
    plot_feature_importance,
    plot_roc_curve   
)

# --------------------------------------------------
# Warning handling (SAFE)
# --------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
DEFAULT_THRESHOLD = 0.317
REPORT_DIR = "reports"


# --------------------------------------------------
# Utility
# --------------------------------------------------
def _ensure_reports():
    os.makedirs(REPORT_DIR, exist_ok=True)


# ==================================================
# 1️⃣ PREDICTION ONLY (NO LABELS – PRODUCTION MODE)
# ==================================================
def predict_only(csv_path, model_path, threshold=DEFAULT_THRESHOLD):
    """
    Used when users upload patient data WITHOUT labels.
    Outputs CAD probability and risk category.
    """

    print("\nLoading input data for prediction...")
    _ensure_reports()

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    if "SUBJECT_ID" not in df.columns:
        raise ValueError("SUBJECT_ID column is required")

    print("Preparing features...")
    X = df.drop(columns=["SUBJECT_ID", "CAD_LABEL"], errors="ignore")

    # Ensure feature alignment
    

    print("Running model inference...")
    probs = model.predict_proba(X)
    preds = (probs >= threshold).astype(int)

    print("Generating prediction results...")
    results = pd.DataFrame({
        "SUBJECT_ID": df["SUBJECT_ID"],
        "CAD_Probability": probs,
        "Risk": ["High" if p >= 0.5 else "Low" for p in probs]
    })

    output_path = os.path.join(REPORT_DIR, "CAD_Predictions.csv")
    results.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print("Prediction completed successfully\n")

    return results


# ==================================================
# 2️⃣ UNSEEN DATA EVALUATION (WITH LABELS)
# ==================================================
def evaluate_unseen(csv_path, model_path, threshold=DEFAULT_THRESHOLD):
    """
    Used to evaluate model generalization on unseen labeled data.
    Prints metrics and generates all evaluation plots.
    """

    print("\nLoading unseen test dataset...")
    _ensure_reports()

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    if "CAD_LABEL" not in df.columns:
        raise ValueError("CAD_LABEL column is required for evaluation")

    if "SUBJECT_ID" not in df.columns:
        raise ValueError("SUBJECT_ID column is required")

    print("Preparing features and labels...")
    X = df.drop(columns=["SUBJECT_ID", "CAD_LABEL"], errors="ignore")
    y = df["CAD_LABEL"].values

    print("Running model inference...")
    probs = model.predict_proba(X)
    preds = (probs >= threshold).astype(int)

    print("Computing evaluation metrics...")
    metrics = {
        "AUC": roc_auc_score(y, probs),
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds)
    }

    print("\nUNSEEN DATA EVALUATION METRICS")
    print("--------------------------------")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")
    print("--------------------------------")

  


    print("Saving evaluation results...")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.join(REPORT_DIR, "CAD_Unseen_Evaluation.csv"),
        index=False
    )

    predictions_df = pd.DataFrame({
        "SUBJECT_ID": df["SUBJECT_ID"],
        "CAD_Probability": probs,
        "Predicted_Label": preds,
        "Actual_Label": y
    })

    predictions_df.to_csv(
        os.path.join(REPORT_DIR, "CAD_Unseen_Predictions.csv"),
        index=False
    )


    # --------------------------------------------------
    # Generate ALL evaluation plots (your graphs)
    # --------------------------------------------------
    print("Generating evaluation plots...")

    stage = "evaluation"

    plot_precision_recall_safety(y, probs, stage)
    plot_threshold_optimization(y, probs, stage)
    plot_decision_buffer_zones(probs, stage)
    plot_confusion_matrix_cost(y, probs, stage)
    plot_performance_targets({
        "Recall": metrics["Recall"],
        "Precision": metrics["Precision"],
        "F1": metrics["F1"],
        "Accuracy": metrics["Accuracy"]
    }, stage)
    plot_roc_curve(y, probs, stage)


    # --------------------------------------------------
    # 🔥 FEATURE IMPORTANCE (FROM LIGHTGBM BASE MODEL)
    # --------------------------------------------------
    base_model = model.get_feature_importance_model()

    plot_feature_importance(
        base_model,
        model.feature_names_,
        stage=stage
    )
    print("All plots saved to reports/figures/evaluation")
    print("Testing Evaluation completed successfully\n")

    return metrics, predictions_df
