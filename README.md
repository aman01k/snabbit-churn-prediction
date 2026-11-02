# Snabbit Customer Churn Prediction

Predict which active customers are likely to make **no booking in the next 60 days**, so the
retention team can act before they leave.

**Status:** full pipeline built and tested on **synthetic data** while waiting for database access.
No result here describes real Snabbit customers. See [docs/project_context.md](docs/project_context.md).

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

| Step | Command / notebook | Output |
|---|---|---|
| 0. Synthetic data | `python src/generate_synthetic_data.py` | `data/raw/synthetic/` |
| 1. Define churn | `notebooks/01_churn_definition.ipynb` | decision: 60 days silent (provisional) |
| 2. Prediction setup & labels | `notebooks/02_prediction_setup.ipynb` | `data/processed/labels.csv` |
| 3. Features | `notebooks/03_feature_engineering.ipynb` | `data/processed/features.csv` |
| 4. Model comparison | `notebooks/04_model_training.ipynb` | choice of model and settings |
| 5. Train final model | `python src/train.py` | `models/churn_model.joblib` |
| 6. Score current customers | `python src/predict.py` | `reports/churn_predictions_<date>.csv` |

Run notebooks from the `notebooks/` folder (`jupyter notebook`), in order. All decisions
(churn window, horizon, excluded services, split dates, model settings, risk cut-offs) live in
[src/config.py](src/config.py).

## Output

`reports/churn_predictions_<date>.csv`, riskiest first:

| Column | Meaning |
|---|---|
| `churn_probability` | Chance of no booking in the next 60 days |
| `risk_category` | High (≥ 50%), Medium (20–50%), Low (< 20%) |
| `days_since_last_booking`, `usual_gap_days` | How silent the customer is vs their normal rhythm |
| `watch_points` | Plain-language flags (e.g. "last booking had a problem"). Simple rules, not an exact model explanation |

## Switching to real data

1. Export customers and bookings into the column layout described in
   [docs/synthetic_data_design.md](docs/synthetic_data_design.md), or adapt `load_bookings` in
   `src/churn_definition.py` and the column notes in `src/features.py`.
2. Point `RAW_DIR` in `src/config.py` at the real extract (company-approved storage only).
3. Re-run notebooks 01 to 04. Every provisional decision must be re-checked on real data.
4. `python src/train.py`, then `python src/predict.py`.

## Layout

```
data/raw/          source data (never committed)
data/processed/    labels and feature tables
docs/              project context and design notes
notebooks/         analysis, one notebook per step
src/               reusable pipeline code
models/            trained model
reports/           prediction outputs
```
