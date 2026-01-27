import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

def evaluate_and_report(model, X, y, report_dir="reports"):
    os.makedirs(f"{report_dir}/figures", exist_ok=True)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    # Metrics
    auc = roc_auc_score(y, probs)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    print("\n📊 MODEL EVALUATION METRICS")
    print("--------------------------------")
    print(f"AUC       : {auc:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("--------------------------------\n")

    # Save metrics CSV
    metrics_df = pd.DataFrame([{
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }])
    metrics_df.to_csv(f"{report_dir}/CAD_Model_Metrics.csv", index=False)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{report_dir}/figures/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    p, r, _ = precision_recall_curve(y, probs)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{report_dir}/figures/precision_recall_curve.png")
    plt.close()
