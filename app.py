
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SlugGuard | Flow Assurance",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 SlugGuard")
st.caption(
    "Explainable AI • Slugging Risk & Mitigation Decision Support"
)
st.warning(
    "Research prototype using simulation data. "
    "Not approved for operational control."
)

# --------------------------------------------------
# 1. Load available trained models
# --------------------------------------------------

MODEL_FILES = {
    "HistGradientBoosting": "slugging_model.joblib",
    "Logistic Regression": "slugging_logisticregression_model.joblib",
}

@st.cache_resource
def load_bundle(path):
    obj = joblib.load(path)

    # Preferred format: saved dictionary bundle
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        features = obj.get("feature_columns")
        label_mode = obj.get("label_mode", "unknown")
    else:
        # Plain estimator: only safe if feature order is known
        model = obj
        features = getattr(model, "feature_names_in_", None)
        label_mode = "unknown"

    if features is None:
        raise ValueError(
            "Expected feature columns were not saved. "
            "Export the model with feature_columns from Colab."
        )

    return {
        "model": model,
        "features": list(features),
        "label_mode": label_mode
    }

available = {
    name: path
    for name, path in MODEL_FILES.items()
    if Path(path).exists()
}

if not available:
    st.error(
        "No model file found. Add slugging_model.joblib "
        "to the same folder as app.py."
    )
    st.stop()

model_name = st.sidebar.selectbox(
    "Select trained model",
    list(available.keys())
)

bundle = load_bundle(available[model_name])
model = bundle["model"]
feature_columns = bundle["features"]

st.sidebar.info(
    f"Target mode: {bundle['label_mode']}"
)

# --------------------------------------------------
# 2. Upload simulation data
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload simulation data (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info(
        "Upload a simulation dataset to begin. "
        "The file must contain the model's required features."
    )
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

st.subheader("1. Input data quality")

st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

missing = [c for c in feature_columns if c not in df.columns]
if missing:
    st.error(
        "Required model features are missing: "
        + ", ".join(missing)
    )
    st.stop()

# Select features in exactly the saved model order
X = df[feature_columns].copy()

# Do not silently replace missing values with zero.
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

all_missing = X.columns[X.isna().all()].tolist()
if all_missing:
    st.error(
        "These model inputs contain no usable numeric values: "
        + ", ".join(all_missing)
    )
    st.stop()

missing_count = int(X.isna().sum().sum())
st.write(f"Missing feature values: {missing_count}")

if missing_count:
    st.warning(
        "Some values are missing. Predictions may be unreliable. "
        "Check the model's approved preprocessing before proceeding."
    )

# --------------------------------------------------
# 3. Run model inference
# --------------------------------------------------

try:
    predicted = model.predict(X)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.stop()

results = df.copy()
results["Predicted_regime"] = predicted

if probabilities is not None:
    classes = list(model.classes_)
    for i, label in enumerate(classes):
        results[f"Score: {label}"] = probabilities[:, i]

# --------------------------------------------------
# 4. Engineering decision-support rules
# --------------------------------------------------

def assessment(regime):
    text = str(regime).lower()

    if "severe" in text or "slugging" in text:
        return (
            "HIGH-PRIORITY REVIEW",
            "Verify pressure and flow evidence, check data quality "
            "and downstream handling constraints, and review the "
            "approved slug-management procedure with the responsible "
            "production or flow assurance engineer."
        )

    if "transitional" in text or "oscillatory" in text:
        return (
            "WATCH / INVESTIGATE",
            "Review pressure and flow variability, choke conditions, "
            "gas-lift conditions where applicable, and operating history."
        )

    if "stable" in text or "continuous" in text:
        return (
            "MONITOR",
            "Continue monitoring and verify that the observed trends "
            "are consistent with the approved operating envelope."
        )

    return (
        "ENGINEERING REVIEW",
        "Confirm the regime definition and consult the responsible engineer."
    )

# --------------------------------------------------
# 5. Dashboard summary
# --------------------------------------------------

latest = results.iloc[-1]
regime = str(latest["Predicted_regime"])
status, recommendation = assessment(regime)

c1, c2, c3 = st.columns(3)
c1.metric("Rows analysed", f"{len(results):,}")
c2.metric("Latest predicted regime", regime)
c3.metric("Decision-support status", status)

st.subheader("2. Engineering recommendation")
st.write(recommendation)

if probabilities is not None:
    score_cols = [
        f"Score: {label}" for label in model.classes_
    ]
    latest_scores = latest[score_cols]

    st.caption(
        "Scores are model outputs, not validated probabilities "
        "of a future slugging event. Calibration has not been assumed."
    )

    score_df = pd.DataFrame({
        "Regime": list(model.classes_),
        "Model score": [
            float(latest[f"Score: {label}"])
            for label in model.classes_
        ]
    })

    fig_scores = px.bar(
        score_df,
        x="Regime",
        y="Model score",
        title="Latest row: model class scores",
        range_y=[0, 1]
    )
    st.plotly_chart(fig_scores, use_container_width=True)

st.subheader("3. Process trends")

preferred = [
    c for c in [
        "TIME_h", "TIME", "PT__OP1-US_CHOKE",
        "PT__OP1-DS_CHOKE", "QLT", "UL", "UG",
        "pressure_drop"
    ]
    if c in results.columns
]

if preferred:
    time_col = "TIME_h" if "TIME_h" in results.columns else (
        "TIME" if "TIME" in results.columns else None
    )

    trend_cols = [
        c for c in preferred if c not in ["TIME_h", "TIME"]
    ]

    if trend_cols:
        plot_data = results.copy()
        if time_col is None:
            plot_data["Sample"] = np.arange(len(plot_data))
            x_col = "Sample"
        else:
            x_col = time_col

        chosen = st.multiselect(
            "Choose variables to display",
            trend_cols,
            default=trend_cols[:2]
        )

        if chosen:
            fig = px.line(
                plot_data,
                x=x_col,
                y=chosen,
                title="Process-variable trends"
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No recognised trend columns found. "
        "Add the correct variable names for your workbook."
    )

st.subheader("4. Prediction history")
st.dataframe(results.tail(100), use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="slugging_predictions.csv",
    mime="text/csv"
)
