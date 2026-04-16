from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "contract_type",
    "payment_method",
    "internet_service",
    "dependents",
    "paperless_billing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn from a CSV file.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with customer features.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/new_customer_predictions.csv",
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--model",
        default="artifacts/churn_model.joblib",
        help="Path to trained model file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    input_path = (project_dir / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (project_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    model_path = (project_dir / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)

    model = joblib.load(model_path)
    data = pd.read_csv(input_path)

    missing_columns = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = data[FEATURE_COLUMNS].copy()
    data["predicted_churn"] = model.predict(features)
    data["predicted_probability"] = model.predict_proba(features)[:, 1]
    data["risk_level"] = pd.cut(
        data["predicted_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
