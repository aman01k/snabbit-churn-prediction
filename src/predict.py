"""Score today's active customers: churn probability, risk group and readable watch points.

Usage:
    python src/predict.py                      # score as of the last date in the data
    python src/predict.py --as-of 2026-10-31

Writes reports/churn_predictions_<date>.csv, riskiest customers first.
"""

import argparse

import joblib
import pandas as pd

import churn_definition as cd
import config
import features
import labeling
import modeling

STALE_AFTER_DAYS = 120
INFREQUENT_GAP_DAYS = 21  # v1 covers frequent services, whose norm is about 5-9 days


def score(bookings, customers, as_of, bundle):
    active = labeling.eligible_customers(bookings, as_of, bundle["excluded_main_services"])
    spine = pd.DataFrame({"customer_id": active.index, "as_of": pd.Timestamp(as_of)})
    table = features.build_features(bookings, customers, spine)

    prob = bundle["model"].predict_proba(table[bundle["features"]])[:, 1]
    out = pd.DataFrame({
        "customer_id": table["customer_id"],
        "churn_probability": prob.round(3),
        "risk_category": modeling.risk_segment(prob, config.RISK_HIGH, config.RISK_MEDIUM),
        "days_since_last_booking": table["recency_days"],
        "usual_gap_days": table["typical_gap_days"].round(0),
        "main_service": table["main_service"],
        "city": table["city"],
        "watch_points": table.apply(watch_points, axis=1),
    })
    return out.sort_values("churn_probability", ascending=False, ignore_index=True)


def watch_points(row):
    """Plain-language flags for the retention team.

    These are simple rules on the features, not an exact explanation of the model's score.
    """
    points = []
    if row["recency_vs_typical_gap"] >= 2:
        points.append(f"overdue: silent {row['recency_vs_typical_gap']:.1f}x usual gap")
    if row["booking_trend_30d"] <= -1:
        points.append("booking less often lately")
    if row["last_booking_bad"]:
        points.append("last booking had a problem")
    if row["expert_cancels_90d"] > 0:
        points.append("expert cancelled recently")
    if row["low_ratings_90d"] > 0:
        points.append("gave a low rating recently")
    if row["n_bookings"] <= 2:
        points.append("only 1-2 bookings so far")
    if row["typical_gap_days"] >= INFREQUENT_GAP_DAYS:
        points.append(f"infrequent booker (about every {row['typical_gap_days']:.0f} days)")
    return "; ".join(points[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", help="scoring date (default: last booking date in the data)")
    args = parser.parse_args()

    bookings = cd.load_bookings(config.RAW_DIR / "bookings.csv")
    customers = pd.read_csv(config.RAW_DIR / "customers.csv", parse_dates=["signup_date"])
    as_of = pd.Timestamp(args.as_of) if args.as_of else bookings["day"].max()
    if as_of > bookings["day"].max():
        raise SystemExit(f"--as-of {as_of.date()} is after the last booking in the data "
                         f"({bookings['day'].max().date()}); recency would be wrong.")

    bundle = joblib.load(config.MODEL_PATH)
    age = (as_of - bundle["trained_through_snapshot"]).days
    if age > STALE_AFTER_DAYS:
        print(f"WARNING: model labels end {age} days before the scoring date; retrain soon.")

    result = score(bookings, customers, as_of, bundle)

    out_path = config.ROOT / "reports" / f"churn_predictions_{as_of.date()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    summary = result.groupby("risk_category", observed=True).agg(
        customers=("customer_id", "size"), avg_probability=("churn_probability", "mean"))
    print(f"scored {len(result):,} active customers as of {as_of.date()}")
    print(summary.loc[["High", "Medium", "Low"]].round(3).to_string())
    print(f"saved {out_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
