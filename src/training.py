import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score

from .model_definition import HarmonizedCADModel
from .evaluation import evaluate_and_report


REPORT_DIR = "reports"
FIG_DIR = os.path.join(REPORT_DIR, "figures", "training")


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_train_vs_validation_accuracy(train_scores, val_scores):
    plt.figure(figsize=(7, 6))
    folds = np.arange(1, len(train_scores) + 1)

    plt.plot(folds, train_scores, marker="o", label="Training Accuracy")
    plt.plot(folds, val_scores, marker="o", label="Validation Accuracy")

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (K-Fold)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "train_vs_validation_accuracy.png"), dpi=300)
    plt.close()


def plot_learning_curve(model_class, X, y):

    print('Generating Learning Curve')
    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model_class(),
        X=X,
        y=y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2)

    plt.plot(train_sizes, val_mean, 'o-', label="Validation Accuracy")
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2)

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve with Confidence Band")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/figures/training/learning_curve.png", dpi=300)
    plt.close()


def train_model(csv_path):

    ensure_dirs()

    print("\nModel training started")
    print("Loading training dataset...")

    df = pd.read_csv(csv_path)

    print("Preparing features and labels...")
    X = df.drop(['CAD_LABEL', 'SUBJECT_ID'], axis=1)
    y = df['CAD_LABEL']

    print("\nRunning Stratified K-Fold Cross Validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_acc_scores = []
    val_acc_scores = []
    train_auc_scores = []
    val_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_cv = HarmonizedCADModel()
        model_cv.fit(X_train, y_train)

        # Training performance
        train_probs = model_cv.predict_proba(X_train)
        train_preds = (train_probs >= model_cv.optimal_threshold).astype(int)

        train_acc_scores.append(accuracy_score(y_train, train_preds))
        train_auc_scores.append(roc_auc_score(y_train, train_probs))

        # Validation performance
        val_probs = model_cv.predict_proba(X_val)
        val_preds = (val_probs >= model_cv.optimal_threshold).astype(int)

        val_acc_scores.append(accuracy_score(y_val, val_preds))
        val_auc_scores.append(roc_auc_score(y_val, val_probs))

    print("\nCross Validation Results")
    print("--------------------------------")
    print(f"Training Accuracy  : {np.mean(train_acc_scores):.4f} ± {np.std(train_acc_scores):.4f}")
    print(f"Validation Accuracy: {np.mean(val_acc_scores):.4f} ± {np.std(val_acc_scores):.4f}")
    print(f"Training AUC       : {np.mean(train_auc_scores):.4f}")
    print(f"Validation AUC     : {np.mean(val_auc_scores):.4f}")
    print("--------------------------------")

    # Save CV results
    cv_results = pd.DataFrame({
        "Train_Accuracy": train_acc_scores,
        "Val_Accuracy": val_acc_scores,
        "Train_AUC": train_auc_scores,
        "Val_AUC": val_auc_scores
    })

    cv_results.to_csv(os.path.join(REPORT_DIR, "cross_validation_results.csv"), index=False)

    # Plot Training vs Validation Accuracy
    plot_train_vs_validation_accuracy(train_acc_scores, val_acc_scores)

    # Plot Learning Curve
    plot_learning_curve(HarmonizedCADModel, X, y)

    print("\nTraining final model on full dataset...")

    model = HarmonizedCADModel()
    model.fit(X, y)

    print("Saving trained model...")
    model.save('models/CAD_Harmonized_91_77_Model.pkl')

    print("Evaluating final model...")
    evaluate_and_report(
        model=model,
        X=X,
        y=y,
        feature_names=model.feature_names_
    )

    print("\nModel training, validation, learning curve, and saving completed.")

    return model
