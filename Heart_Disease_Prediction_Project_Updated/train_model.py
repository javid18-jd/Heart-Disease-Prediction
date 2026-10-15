"""Download the processed Cleveland UCI dataset and train a Random Forest model.

The dataset is for education/research. It is small and is not validated for clinical use.
"""
from pathlib import Path
import json
import urllib.request

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = DATA_DIR / "processed.cleveland.data"
MODEL_PATH = MODEL_DIR / "heart_disease_model.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "thalach", "fbs", "exang", "oldpeak", "slope"]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print("Downloading the UCI Cleveland Heart Disease dataset...")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                DATA_PATH.write_bytes(response.read())
        except Exception as exc:
            raise SystemExit(
                "Dataset download failed. Check your internet connection, then retry. "
                f"Source: {URL}\\nDetails: {exc}"
            )

    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["num"])
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    if len(df) < 100:
        raise SystemExit("Dataset appears incomplete. Delete data/processed.cleveland.data and rerun.")

    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset": "UCI Heart Disease - processed Cleveland subset",
        "rows_after_target_cleaning": int(len(df)),
        "features": FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "warning": "Small historical dataset; metrics are for demonstration only, not clinical validation."
    }
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dataset rows used: {len(df)}")
    print(f"Test accuracy: {metrics['accuracy']:.3f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
