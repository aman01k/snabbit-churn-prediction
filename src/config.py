"""Project-wide settings. Change decisions here, not inside the pipeline code.

Values marked PROVISIONAL were chosen on synthetic data and must be confirmed on real
data and with the Snabbit business team.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "synthetic"  # point at the real extract later
PROCESSED_DIR = ROOT / "data" / "processed"

# Churn definition (notebooks/01_churn_definition.ipynb). PROVISIONAL.
CHURN_INACTIVITY_DAYS = 60

# Prediction target: "will this active customer make no booking in the next N days?"
# Set equal to the churn window, so a positive label means they become churned. PROVISIONAL:
# depends on how early the retention team needs to act.
HORIZON_DAYS = 60

# Months of booking history required before the first snapshot, so features have data.
MIN_HISTORY_DAYS = 90

# Customers whose main service is booked so rarely that "no booking in 60 days" is normal
# behaviour. In notebook 02, most of their positive labels came back later. Version 1 of
# the model covers frequent services only. PROVISIONAL, a business decision.
EXCLUDED_MAIN_SERVICES = ["deep_cleaning", "bathroom_cleaning"]

# Time-based split on snapshot dates. Each boundary leaves a gap of at least
# HORIZON_DAYS so a training label never looks into the validation/test period.
# (Snapshots falling in a gap are not used.)
TRAIN_END = pd.Timestamp("2026-03-01")  # train: snapshots <= this
VALID_DATES = [pd.Timestamp("2026-05-01")]
TEST_START = pd.Timestamp("2026-07-01")  # test: snapshots >= this (with a full label window)

# Chosen model (notebook 04): gradient boosting, settings picked on the validation snapshot.
MODEL_PARAMS = {"learning_rate": 0.03, "max_iter": 500, "max_leaf_nodes": 15, "min_samples_leaf": 100}
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

# Risk bands on predicted probability. PROVISIONAL: should match retention-team capacity.
RISK_HIGH = 0.50
RISK_MEDIUM = 0.20
