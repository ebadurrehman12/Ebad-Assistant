from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train customer churn model.")
    parser.add_argument(
        "--data-path",
        default="",
        help="Optional CSV path (e.g., Kaggle Telco churn dataset). If omitted, synthetic data is used.",
    )
    return parser.parse_args()


def generate_customer_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic churn dataset."""
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=n_customers)
    monthly_charges = np.round(rng.normal(loc=75, scale=25, size=n_customers).clip(20, 150), 2)
    support_calls = rng.integers(0, 8, size=n_customers)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_customers)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.4, 0.2, 0.2, 0.2],
        size=n_customers,
    )
    internet_service = rng.choice(["Fiber optic", "DSL", "No"], p=[0.5, 0.35, 0.15], size=n_customers)
    dependents = rng.choice(["Yes", "No"], p=[0.35, 0.65], size=n_customers)
    paperless_billing = rng.choice(["Yes", "No"], p=[0.7, 0.3], size=n_customers)

    total_charges = np.round((tenure_months * monthly_charges) + rng.normal(0, 50, n_customers), 2).clip(20, None)

    # Simulated churn logic to mimic real behavior patterns.
    churn_score = (
        1.4 * (contract_type == "Month-to-month")
        + 1.1 * (payment_method == "Electronic check")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.25 * support_calls
        - 0.015 * tenure_months
        - 0.008 * total_charges
        + 0.7 * (paperless_billing == "Yes")
        + rng.normal(0, 0.7, n_customers)
    )

    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = (rng.random(n_customers) < churn_probability).astype(int)

    return pd.DataFrame(
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
            "churn": churn,
        }
    )


def load_telco_kaggle_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and clean the common Telco churn Kaggle dataset schema."""
    df = pd.read_csv(csv_path)

    # Standard dataset cleanup (WA_Fn-UseC_-Telco-Customer-Churn.csv style).
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns:
        df["churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
        df = df.drop(columns=["Churn"])
    elif "churn" in df.columns:
        if df["churn"].dtype == object:
            df["churn"] = df["churn"].map({"Yes": 1, "No": 0}).fillna(df["churn"]).astype(int)
    else:
        raise ValueError("Target column not found. Expected 'Churn' or 'churn'.")

    return df


def build_and_evaluate_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Train pipeline and return evaluation metrics."""
    target_col = "churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "churn_rate": float(y.mean()),
    }

    test_predictions = X_test.copy()
    test_predictions["actual_churn"] = y_test.values
    test_predictions["predicted_churn"] = y_pred
    test_predictions["predicted_probability"] = y_proba

    return pipeline, metrics, test_predictions


def save_outputs(model: Pipeline, metrics: dict, test_predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "churn_model.joblib"
    json_path = output_dir / "results.json"
    text_path = output_dir / "results.txt"
    predictions_path = output_dir / "test_predictions.csv"

    joblib.dump(model, model_path)
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_predictions.to_csv(predictions_path, index=False)

    lines = [
        "Customer Churn Model Results",
        "=" * 30,
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 Score: {metrics['f1_score']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"Train Size: {metrics['train_size']}",
        f"Test Size: {metrics['test_size']}",
        f"Churn Rate: {metrics['churn_rate']:.4f}",
        f"Confusion Matrix: {metrics['confusion_matrix']}",
    ]
    text_path.write_text("\n".join(lines), encoding="utf-8")


def save_feature_importance(model: Pipeline, output_dir: Path, top_n: int = 15) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "feature_importance.csv"
    feature_plot_path = output_dir / "feature_importance.png"

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    importance_df.to_csv(feature_path, index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title("Top Feature Importance (Why Churn)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(feature_plot_path, dpi=150)
    plt.close()


def save_visualizations(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_plot_path = output_dir / "metrics_bar.png"
    confusion_plot_path = output_dir / "confusion_matrix.png"

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_values = [metrics[name] for name in metric_names]
    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1)
    plt.title("Customer Churn Model Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=150)
    plt.close()

    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    plt.tight_layout()
    plt.savefig(confusion_plot_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = (project_dir / data_path).resolve()
        df = load_telco_kaggle_dataset(data_path)
    else:
        df = generate_customer_churn_data()

    model, metrics, test_predictions = build_and_evaluate_model(df)
    output_dir = project_dir / "artifacts"
    save_outputs(model, metrics, test_predictions, output_dir)
    save_visualizations(metrics, output_dir)
    save_feature_importance(model, output_dir)

    print("\nCustomer Churn Project Completed")
    print("-" * 35)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nSaved files in artifacts/:")
    print("- churn_model.joblib")
    print("- results.json")
    print("- results.txt")
    print("- test_predictions.csv")
    print("- metrics_bar.png")
    print("- confusion_matrix.png")
    print("- feature_importance.csv")
    print("- feature_importance.png")


if __name__ == "__main__":
    main()
