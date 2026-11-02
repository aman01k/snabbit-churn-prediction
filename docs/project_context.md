# Project Context

Handover summary so any future session (or teammate) can pick this up.

## Goal

Predict which Snabbit customers are likely to stop booking (cleaning, dishwashing, laundry,
etc.), give each a churn probability and a High / Medium / Low risk group, and turn that into
retention recommendations. Customers rarely cancel an account; churn means they *stop
booking*. We predict **future** churn, not list customers who already left.

## Status (2 Nov 2026)

- No access to Snabbit data yet. We're building the pipeline on synthetic data
  ([synthetic_data_design.md](synthetic_data_design.md)) and will swap in real data later.
- Environment: local Python 3.14 virtualenv in `.venv` on the company MacBook.
- Churn definition analysis done on synthetic data (`notebooks/01_churn_definition.ipynb`):
  provisional pick is 60 days without a booking for frequent services. A relative rule
  (3× the customer's usual gap) works better for infrequent services and is planned for v2.
  Must be redone on real data.
- Prediction setup (`notebooks/02_prediction_setup.ipynb`, settings in `src/config.py`):
  predict "no booking in next 60 days" for active customers at monthly snapshots; time split
  with 60-day buffers; v1 excludes deep/bathroom cleaning customers. Labels saved to
  `data/processed/labels.csv`.
- Features (`notebooks/03_feature_engineering.ipynb`, code in `src/features.py`): 29 point-in-time
  features with a passing leakage test, saved to `data/processed/features.csv`.
- Modelling (`notebooks/04_model_training.ipynb`, code in `src/modeling.py`, `src/train.py`):
  gradient boosting (scikit-learn HistGradientBoosting; XGBoost needs Homebrew's libomp, not
  installed). Test ROC-AUC 0.91 / PR-AUC 0.74 on synthetic data, well calibrated. Model saved to
  `models/churn_model.joblib` (production fit on all labeled snapshots; honest test scores stored inside).
- Scoring (`src/predict.py`): scores active customers as of a date, writes
  `reports/churn_predictions_<date>.csv` with probability, risk group and watch points.
  End-to-end pipeline complete on synthetic data.

## Not decided yet (don't assume)

Churn definition (30/45/60-day inactivity are only hypotheses), prediction horizon,
the real schema and which columns exist, database technology, how we get data access,
data-security rules, deployment, and whether this is a one-time analysis or a recurring
pipeline.

## Working principles

- Data understanding and the churn definition come before any model training.
- Features may only use information available on the prediction date (no leakage).
- Split data by time; handle class imbalance; look at precision/recall and calibration,
  not just accuracy.
- Never invent column names or results.
- Company data stays on company-approved storage. No personal Colab or Drive without
  authorization.

## Questions for Snabbit once access arrives

1. Which database, and will I get SQL access?
2. Which tables exist (customers, bookings, payments, cancellations, ratings, support)?
3. How much history is there?
4. Is there an existing churn definition or analysis ("lapsed users" etc.)?
5. One-time analysis, or a recurring prediction system?
6. What restrictions apply to cloud tools?
