"""
Training script: builds a binary classifier to predict whether a requested credit amount is HIGH (> dataset median).
Dataset: data/german_credit_data.csv
Target (proxy): HighAmount = (Credit amount > median)
Outputs: artifacts/model.joblib, artifacts/metrics.json, artifacts/schema.json, plots
"""

import json
import joblib
import pandas as pd
import numpy  # משתמשים בשם המלא 'numpy' ולא ב-np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import itertools

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "german_credit_data.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True, parents=True)


def main():
    # Load data
    df = pd.read_csv(DATA)

    # Create proxy target: HighAmount
    median_amt = df["Credit amount"].median()
    df["HighAmount"] = (df["Credit amount"] > median_amt).astype(int)

    # Drop non-features
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    target = "HighAmount"
    X = df.drop(columns=[target, "Credit amount"])
    y = df[target]

    # Feature types (שימי לב: numpy.number, לא np.number)
    numeric_features = X.select_dtypes(include=[numpy.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[numpy.number]).columns.tolist()

    # Preprocessing
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ])

    # Model
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("preprocess", preprocess), ("model", clf)])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "threshold": 0.5,
        "target_definition": f"HighAmount = (Credit amount > median {median_amt})"
    }

    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save artifacts
    joblib.dump(pipe, ART / "model.joblib")

    with open(ART / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)

    schema = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": X.columns.tolist(),
        "target": target
    }
    with open(ART / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # ROC Curve
    fig = plt.figure()
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(ART / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    # Confusion Matrix
    cm_array = numpy.array(cm)
    fig2 = plt.figure()
    plt.imshow(cm_array, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = numpy.arange(2)
    plt.xticks(tick_marks, ["Low", "High"], rotation=45)
    plt.yticks(tick_marks, ["Low", "High"])
    for i, j in itertools.product(range(cm_array.shape[0]), range(cm_array.shape[1])):
        plt.text(j, i, cm_array[i, j], horizontalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(ART / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig2)

    print("✅ Saved artifacts to", ART.resolve())
    print("📊 Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
