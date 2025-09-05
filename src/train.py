# src/train.py
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -------- paths --------
DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip spaces & fix TotalCharges which arrives as string with blanks
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    # Target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    # Drop leakage / IDs
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df

def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"].values
    return X, y

def define_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify numeric & categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def try_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42
        )
    except Exception:
        return None

def pr_auc_score(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def main():
    print("Loading & cleaning data...")
    df = load_clean_data(DATA_PATH)
    X, y = split_xy(df)

    print("Train/val/test split...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("Defining preprocessing...")
    preprocessor = define_preprocessor(X)

    # Base pipeline with a placeholder classifier
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None))
    ])

    # Two grids: Logistic Regression & Random Forest
    param_grid = [
        {
            "clf": [LogisticRegression(max_iter=1000, class_weight="balanced")],
            "clf__C": [0.1, 1.0, 3.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"]
        },
        {
            "clf": [RandomForestClassifier(random_state=42, class_weight="balanced")],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 8, 12],
            "clf__min_samples_split": [2, 10]
        }
    ]

    # Optionally add XGBoost if available in the environment
    xgb = try_xgboost()
    if xgb is not None:
        param_grid.append({
            "clf": [xgb],
            # you can tune fewer params to keep it fast
        })

    print("Grid search… (this uses cross-validation)")
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)

    print(f"Best estimator: {gs.best_estimator_}")
    print(f"Best CV ROC-AUC: {gs.best_score_:.4f}")

    # Evaluate on validation, then final on test
    best = gs.best_estimator_

    def eval_split(split_name, Xs, ys):
        probs = best.predict_proba(Xs)[:, 1]
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(ys, preds)),
            "f1": float(f1_score(ys, preds)),
            "roc_auc": float(roc_auc_score(ys, probs)),
            "pr_auc": float(pr_auc_score(ys, probs)),
        }
        print(f"\n=== {split_name} Metrics ===")
        print(metrics)
        print("\nClassification Report")
        print(classification_report(ys, preds, digits=4))
        return metrics, preds, probs

    val_metrics, _, _ = eval_split("VALID", X_valid, y_valid)
    test_metrics, test_preds, test_probs = eval_split("TEST", X_test, y_test)

    # Confusion matrix (test)
    cm = confusion_matrix(y_test, (test_probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrix_test.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    # Save artifacts
    print("Saving artifacts…")
    dump(best, MODEL_DIR / "churn_pipeline.pkl")

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({
            "cv_best_roc_auc": float(gs.best_score_),
            "valid": val_metrics,
            "test": test_metrics
        }, f, indent=2)

    # Save feature columns after fit (for future UI mapping/logging)
    # We store original columns and dtypes (useful for Streamlit form on Day 2)
    feature_cols = {c: str(X[c].dtype) for c in X.columns}
    with open(REPORT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\nAll done. Artifacts:")
    print(f"- models/churn_pipeline.pkl")
    print(f"- reports/metrics.json")
    print(f"- reports/confusion_matrix_test.png")
    print(f"- reports/feature_columns.json")

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place the CSV there.")
    main()
