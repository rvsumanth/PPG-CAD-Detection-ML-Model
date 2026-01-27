import os
import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*"
)




# -------------------------------------------------
# 1️⃣ PURE PREDICTION (NO LABELS)
# -------------------------------------------------
def predict_only(csv_path, model_path, output_dir="reports"):
    print("\nLoading input data for prediction...")
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    if "SUBJECT_ID" not in df.columns:
        raise ValueError("SUBJECT_ID column is required")

    
    print("Preparing features...")
    X = df.drop(columns=["SUBJECT_ID", "CAD_LABEL"], errors="ignore")
    X = X[model.feature_names_]

    
    print("Running model inference...")
    probs = model.predict_proba(X)
    preds = model.predict(X)

    results = pd.DataFrame({
        "SUBJECT_ID": df["SUBJECT_ID"],
        "CAD_Probability": probs,
        "Risk": ["High" if p >= model.optimal_threshold else "Low" for p in probs]
    })

    output_path = os.path.join(output_dir, "CAD_Predictions.csv")
    results.to_csv(output_path, index=False)

    print("✅ Prediction completed (NO labels)")
    print(f"📁 Saved to: {output_path}")

    return results


# -------------------------------------------------
# 2️⃣ UNSEEN DATA EVALUATION (WITH LABELS)
# -------------------------------------------------
def evaluate_unseen(csv_path, model_path, output_dir="reports"):
    print("\nLoading unseen test dataset...")
    os.makedirs(output_dir, exist_ok=True)

    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)

    if "CAD_LABEL" not in df.columns:
        raise ValueError("CAD_LABEL column is required for evaluation")
    print("Preparing features and labels...")
    X = df.drop(columns=["SUBJECT_ID", "CAD_LABEL"], errors="ignore")
    X = X[model.feature_names_]
    y = df["CAD_LABEL"]

    print("Running model inference...")
    probs = model.predict_proba(X)
    preds = model.predict(X)

    print("Evaluating unseen data performance...")
    metrics = {
        "AUC": roc_auc_score(y, probs),
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds)
    }

    print("\n📊 UNSEEN DATA EVALUATION METRICS")
    print("--------------------------------")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")
    print("--------------------------------")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "CAD_Unseen_Evaluation.csv"),
        index=False
    )

    # Save predictions
    predictions = pd.DataFrame({
        "SUBJECT_ID": df["SUBJECT_ID"],
        "CAD_Probability": probs,
        "Predicted_Label": preds,
        "Actual_Label": y
    })

    predictions.to_csv(
        os.path.join(output_dir, "CAD_Unseen_Predictions.csv"),
        index=False
    )

    print("✅ Evaluation + predictions saved")

    return metrics, predictions
