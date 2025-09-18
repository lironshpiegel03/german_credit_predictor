import streamlit as st
import pandas as pd, json, joblib, os
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay,
    precision_score, recall_score, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "german_credit_data.csv"
ART  = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="German Credit — High Amount Predictor", layout="wide")
st.title("💳 German Credit — High Amount Predictor")

# ---------- Helpers ----------
def build_target(df: pd.DataFrame):
    median_amt = df["Credit amount"].median()
    df = df.copy()
    df["HighAmount"] = (df["Credit amount"] > median_amt).astype(int)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    target = "HighAmount"
    X = df.drop(columns=[target, "Credit amount"])
    y = df[target]
    return X, y, target, median_amt, df

def make_preprocess(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
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
    schema = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": X.columns.tolist(),
        "target": "HighAmount"
    }
    return preprocess, schema

def train_and_eval(model_kind: str, test_size: float, random_state: int,
                   rf_n_estimators: int = 200, rf_max_depth: int | None = None):
    df = pd.read_csv(DATA)
    X, y, target, median_amt, _df_full = build_target(df)
    preprocess, schema = make_preprocess(X)

    if model_kind == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            n_jobs=-1,
            random_state=random_state
        )

    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "threshold": 0.5,
        "target_definition": f"HighAmount = (Credit amount > median {median_amt})",
        "model": model_kind
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    # save artifacts
    joblib.dump(pipe, ART / "model.joblib")
    with open(ART / "schema.json","w") as f: json.dump(schema, f, indent=2)
    with open(ART / "metrics.json","w") as f: json.dump({"metrics":metrics,"confusion_matrix":cm}, f, indent=2)

    # plots (static for 0.5)
    fig = plt.figure()
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("ROC Curve")
    (ART / "roc_curve.png").unlink(missing_ok=True)
    plt.savefig(ART / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    cm_array = np.array(cm)
    fig2 = plt.figure()
    plt.imshow(cm_array, interpolation='nearest')
    plt.title("Confusion Matrix (threshold = 0.5)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Low","High"], rotation=45)
    plt.yticks(tick_marks, ["Low","High"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_array[i, j], ha="center", va="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    (ART / "confusion_matrix.png").unlink(missing_ok=True)
    plt.savefig(ART / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig2)

    return pipe, schema, metrics

@st.cache_resource
def load_artifacts():
    model = joblib.load(ART/"model.joblib")
    with open(ART/"schema.json") as f: schema = json.load(f)
    with open(ART/"metrics.json") as f: metrics = json.load(f)
    return model, schema, metrics

# ------------- Sidebar: Retrain + Threshold -------------
with st.sidebar:
    st.header("⚙️ Retrain / Settings")
    model_kind = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)

    if model_kind == "Random Forest":
        rf_n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        rf_max_depth = st.selectbox("max_depth (None = unlimited)", [None, 3, 5, 7, 10, 15, 20], index=0)
    else:
        rf_n_estimators, rf_max_depth = None, None

    threshold = st.slider("Decision threshold (High if proba ≥ threshold)", 0.05, 0.95, 0.50, 0.01)

    if st.button("🔁 Retrain now"):
        _ = train_and_eval(
            model_kind=model_kind,
            test_size=test_size,
            random_state=int(random_state),
            rf_n_estimators=rf_n_estimators if rf_n_estimators is not None else 200,
            rf_max_depth=rf_max_depth if rf_max_depth is not None else None
        )
        st.success("Model retrained and artifacts updated ✔")
        st.cache_resource.clear()  # reload below

# ------------- Main area -------------
model, schema, all_metrics = load_artifacts()
m = all_metrics["metrics"]

with st.expander("ℹ️ Project metrics and info", expanded=True):
    st.json({
        "model": m.get("model", "Logistic Regression"),
        "accuracy": m["accuracy"],
        "f1": m["f1"],
        "roc_auc": m["roc_auc"],
        "n_train": m["n_train"],
        "n_test": m["n_test"],
        "target_definition": m["target_definition"],
    })
    c1, c2 = st.columns(2)
    with c1:
        st.image(ART/"roc_curve.png", caption="ROC Curve")
    with c2:
        st.image(ART/"confusion_matrix.png", caption="Confusion Matrix (threshold=0.5)")

# ---------- Dynamic evaluation by Threshold ----------
st.subheader("📈 Evaluation on Holdout (Dynamic by Threshold)")

# Fixed split for visualization (same seed as train.py's default)
df_all = pd.read_csv(DATA)
X_all, y_all, _, _, _ = build_target(df_all)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

proba_test = model.predict_proba(X_test_h)[:, 1]
y_pred_thr = (proba_test >= threshold).astype(int)

prec = precision_score(y_test_h, y_pred_thr)
rec  = recall_score(y_test_h, y_pred_thr)
f1   = f1_score(y_test_h, y_pred_thr)

m1, m2, m3 = st.columns(3)
m1.metric("Precision", f"{prec:.3f}")
m2.metric("Recall",    f"{rec:.3f}")
m3.metric("F1",        f"{f1:.3f}")

cm_dyn = confusion_matrix(y_test_h, y_pred_thr)
fig_dyn = plt.figure()
plt.imshow(cm_dyn, interpolation='nearest')
plt.title(f"Confusion Matrix (threshold = {threshold:.2f})")
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ["Low","High"], rotation=45)
plt.yticks(ticks, ["Low","High"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_dyn[i, j], ha="center", va="center")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
st.pyplot(fig_dyn)

precisions, recalls, thrs = precision_recall_curve(y_test_h, proba_test)
ap = average_precision_score(y_test_h, proba_test)

fig_pr = plt.figure()
plt.plot(recalls, precisions)
plt.title(f"Precision-Recall Curve (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
# mark current threshold point
if len(thrs) > 0:
    idx = (np.abs(thrs - threshold)).argmin()
    if idx < len(precisions):
        plt.scatter([recalls[idx]], [precisions[idx]])
        plt.annotate(f"thr={threshold:.2f}", (recalls[idx], precisions[idx]))
st.pyplot(fig_pr)

# ---------- Predict a single case ----------
st.header("🔮 Predict a single case")
cols = st.columns(2)
df_raw = pd.read_csv(DATA)

# force integers for these features even if dtype is float
force_int_features = {"Age", "Duration", "Job"}

def is_integer_like(series: pd.Series) -> bool:
    s = series.dropna()
    return pd.api.types.is_integer_dtype(series) or ((s % 1) == 0).all()

inputs = {}
for i, feat in enumerate(schema["all_features"]):
    col = cols[i % 2]
    if feat in schema["numeric_features"]:
        s = df_raw[feat].dropna()
        min_v = float(s.min()) if not s.empty else 0.0
        max_v = float(s.max()) if not s.empty else 100.0

        # כופים שלמים ב-Age/Duration/Job וגם לכל עמודה שמזוהה integer-like
        if (feat in force_int_features) or is_integer_like(df_raw[feat]):
            default_val = int(round(s.median())) if not s.empty else 0
            inputs[feat] = col.number_input(
                feat,
                value=default_val,
                min_value=int(min_v),
                max_value=int(max_v),
                step=1,
                format="%d"
            )
        else:
            default_val = float(s.median()) if not s.empty else 0.0
            inputs[feat] = col.number_input(
                feat,
                value=default_val,
                min_value=min_v,
                max_value=max_v,
                step=0.1
            )
    else:
        options = sorted(df_raw[feat].dropna().unique().tolist())
        default_idx = 0 if options else None
        inputs[feat] = col.selectbox(feat, options=options, index=default_idx)

if st.button("Predict"):
    X_input = pd.DataFrame([inputs])
    proba = model.predict_proba(X_input)[0,1]
    pred = int(proba >= threshold)  # uses the threshold slider
    st.success(f"Prediction: **{'High' if pred==1 else 'Low'}** | Probability High = **{proba:.3f}** | Threshold = {threshold:.2f}")

# ---------- Batch predictions ----------
st.header("📤 Batch predictions from CSV")
uploaded = st.file_uploader("Upload a CSV with the following columns:", type=["csv"])
st.write(schema["all_features"])
if uploaded is not None:
    new_df = pd.read_csv(uploaded)
    missing = [c for c in schema["all_features"] if c not in new_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        probs = model.predict_proba(new_df)[:,1]
        preds = (probs >= threshold).astype(int)
        out = new_df.copy()
        out["proba_high"] = probs
        out["pred_high"] = preds
        st.dataframe(out.head())
        st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

st.caption("Retrain, tune hyperparameters, and adjust the decision threshold. Integer-only inputs enforced for Age/Duration/Job.")
