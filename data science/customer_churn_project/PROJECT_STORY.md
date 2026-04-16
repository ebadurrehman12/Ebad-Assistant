# Customer Churn Prediction Story

## Project Overview

This project predicts customer churn (whether a customer is likely to leave a service) using machine learning.

The core objective is to help subscription businesses reduce revenue loss by identifying at-risk customers early and enabling targeted retention actions.

## Problem Statement

In telecom and subscription businesses, churn directly impacts growth and profitability.

Without a churn prediction system:
- Retention teams act too late
- Marketing budgets are wasted on untargeted campaigns
- High-value users may leave silently

This project solves that by assigning churn risk probabilities for each customer.

## Business Context (Pakistan-Market Framing)

The approach is adaptable for Pakistan-focused telecom/subscription use cases where:
- Price sensitivity can be high
- Contract type and payment method strongly influence churn behavior
- Service quality and support experience drive customer loyalty

This repository currently uses:
- Synthetic dataset (for full reproducibility), and/or
- Public Kaggle telecom churn dataset (for real-world benchmarking)

It is intentionally built with public/anonymized data workflows for legal and ethical compliance.

## Data and Features

Model-relevant features include:
- Tenure
- Monthly charges
- Total charges
- Number of support calls
- Contract type
- Payment method
- Internet/service type
- Dependents
- Paperless billing

Target:
- `churn = 1` (customer likely to leave)
- `churn = 0` (customer likely to stay)

## Modeling Approach

- Preprocessing:
  - Standard scaling for numeric features
  - One-hot encoding for categorical features
- Model:
  - Random Forest Classifier
- Evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion matrix

## Results (Current Run)

- Accuracy: `0.9630`
- Precision: `0.7229`
- Recall: `0.8108`
- F1 Score: `0.7643`
- ROC-AUC: `0.9836`

Interpretation:
- The model separates churn/non-churn patterns very well (high ROC-AUC).
- Recall is strong, meaning the system captures many real churners.
- Precision is moderate-good, meaning most flagged customers are relevant.

## Why This Is Useful

The output enables practical retention operations:
- Prioritize outreach to high-risk users
- Offer loyalty plans/discounts only where needed
- Reduce churn-driven revenue leakage
- Improve customer lifetime value (CLV)

## Explainability (Why Churn)

The project includes feature importance outputs:
- `artifacts/feature_importance.csv`
- `artifacts/feature_importance.png`

These help explain which attributes influence churn prediction most, making the model easier to trust for business teams.

## Live and Batch Delivery

- Batch scoring script: `predict_churn.py`
- Interactive dashboard: `dashboard.py` (Streamlit)
- Saved artifacts:
  - model
  - metrics
  - confusion matrix chart
  - feature importance chart
  - customer-level prediction CSV

## Ethical and Compliance Notes

This project avoids private or unauthorized data collection workflows.
Preferred data sources are public, licensed, and anonymized datasets.

Why:
- Reduces legal and policy risk
- Improves reproducibility and credibility on GitHub
- Aligns with responsible AI and data privacy practices

## How To Run (Quick)

```bash
pip install -r requirements.txt
python churn_project.py
python predict_churn.py --input sample_new_customers.csv
python -m streamlit run dashboard.py
```

For Kaggle data:

```bash
python churn_project.py --data-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Portfolio Value

This repository demonstrates end-to-end ML delivery:
- Problem framing
- Data preparation
- Model training and evaluation
- Explainability
- Batch prediction
- Live dashboard communication

It is a production-style starter for churn analytics projects in telecom and subscription businesses.
