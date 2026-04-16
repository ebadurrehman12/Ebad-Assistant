from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def load_assets():
    model = joblib.load(ARTIFACTS_DIR / "churn_model.joblib")
    metrics = json.loads((ARTIFACTS_DIR / "results.json").read_text(encoding="utf-8"))
    test_predictions = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    feature_importance = None
    feature_importance_path = ARTIFACTS_DIR / "feature_importance.csv"
    if feature_importance_path.exists():
        feature_importance = pd.read_csv(feature_importance_path)
    return model, metrics, test_predictions, feature_importance


def run() -> None:
    st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
    st.title("Customer Churn Live Dashboard")
    st.caption("Interactive prediction and model performance view")

    if not (ARTIFACTS_DIR / "churn_model.joblib").exists():
        st.warning("Model artifacts not found. Run: python churn_project.py")
        return

    model, metrics, test_predictions, feature_importance = load_assets()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.subheader("Prediction Probability Distribution")
    st.bar_chart(test_predictions["predicted_probability"], use_container_width=True)

    if feature_importance is not None and not feature_importance.empty:
        st.subheader("Why Customers Churn (Top Features)")
        chart_df = feature_importance.set_index("feature")["importance"]
        st.bar_chart(chart_df, use_container_width=True)

    st.subheader("Live Single-Customer Prediction")
    left, right = st.columns(2)

    with left:
        tenure_months = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges", 20.0, 150.0, 85.0)
        total_charges = st.slider("Total Charges", 20.0, 10000.0, 1000.0)
        support_calls = st.slider("Support Calls", 0, 7, 2)
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with right:
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    sample = pd.DataFrame(
        [
            {
                "tenure_months": tenure_months,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "support_calls": support_calls,
                "contract_type": contract_type,
                "payment_method": payment_method,
                "internet_service": internet_service,
                "dependents": dependents,
                "paperless_billing": paperless_billing,
            }
        ]
    )

    if st.button("Predict Churn", type="primary"):
        churn_proba = model.predict_proba(sample)[0, 1]
        churn_label = int(churn_proba >= 0.5)
        st.success(f"Predicted churn probability: {churn_proba:.2%}")
        st.info(f"Predicted class: {'Churn' if churn_label == 1 else 'No Churn'}")

    st.subheader("Recent Test Predictions")
    st.dataframe(test_predictions.head(20), use_container_width=True)


if __name__ == "__main__":
    run()
