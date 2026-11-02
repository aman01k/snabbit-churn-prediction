"""Evaluate the chosen churn model on the held-out test split, then retrain it on all labels and save.

Two fits, on purpose:
1. Evaluation fit: train + valid snapshots, scored once on the untouched test snapshots.
   This gives the honest performance estimate.
2. Production fit: same settings on every labeled snapshot, so the saved model has also
   learned from the most recent months. Its quality is the one estimated in step 1.

Usage:
    python src/train.py
"""

import joblib
import pandas as pd

import config
import modeling


def train_final(table):
    """Fit on train+valid and score the test split. Returns (model, test metrics, last snapshot used)."""
    fit_rows = table[table["split"].isin(["train", "valid"])]
    test = table[table["split"] == "test"]

    model = modeling.make_gradient_boosting(**config.MODEL_PARAMS)
    model.fit(fit_rows[modeling.FEATURES], fit_rows[modeling.TARGET])

    prob = model.predict_proba(test[modeling.FEATURES])[:, 1]
    metrics = modeling.evaluate(test[modeling.TARGET], prob)
    return model, metrics, fit_rows["as_of"].max()


def train_production(table):
    """Fit the same model on every labeled snapshot (gaps were only needed for honest evaluation)."""
    model = modeling.make_gradient_boosting(**config.MODEL_PARAMS)
    model.fit(table[modeling.FEATURES], table[modeling.TARGET])
    return model, table["as_of"].max()


def main():
    table = pd.read_csv(config.PROCESSED_DIR / "features.csv", parse_dates=["as_of"])

    _, metrics, _ = train_final(table)
    print("held-out test performance:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    model, trained_through = train_production(table)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": modeling.FEATURES,
        "params": config.MODEL_PARAMS,
        "trained_through_snapshot": trained_through,
        "churn_inactivity_days": config.CHURN_INACTIVITY_DAYS,
        "horizon_days": config.HORIZON_DAYS,
        "excluded_main_services": config.EXCLUDED_MAIN_SERVICES,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
    }, config.MODEL_PATH)
    print(f"saved {config.MODEL_PATH.relative_to(config.ROOT)} "
          f"(production fit on all labels through {trained_through.date()})")


if __name__ == "__main__":
    main()
