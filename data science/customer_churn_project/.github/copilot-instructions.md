# Copilot Instructions for This Repository

Use these conventions when generating code:

1. Keep code Python 3.11+ compatible.
2. Avoid adding non-essential dependencies.
3. Preserve the existing churn pipeline pattern:
   - preprocessing in `ColumnTransformer`
   - model in scikit-learn `Pipeline`
4. Keep outputs in `artifacts/` and do not hardcode absolute paths.
5. Prefer public/anonymized datasets and avoid private scraping workflows.
6. If adding a feature:
   - update `README.md`
   - update `PROJECT_STORY.md` if business impact changes
7. For dashboard updates, keep Streamlit startup command:
   - `python -m streamlit run dashboard.py`
