"""Point-in-time customer features for each (customer_id, as_of) row of the label table.

The one rule: a feature for snapshot `as_of` may only use bookings on or before `as_of`.
Every function here filters by date first, so later code can't accidentally see the future.

Booking columns used (map real Snabbit columns onto these when data arrives):
    customer_id, booking_datetime, service_type, status, list_price, discount_amount,
    amount_paid, coupon_used, expert_delay_minutes, rating, had_complaint
Customer columns used: customer_id, signup_date, city, acquisition_channel
"""

import numpy as np
import pandas as pd

import churn_definition as cd

LATE_MINUTES = 30
LOW_RATING = 2


def build_features(bookings, customers, spine):
    """Attach features to `spine` (a label table with customer_id and as_of columns)."""
    parts = []
    for as_of, rows in spine.groupby("as_of"):
        parts.append(snapshot_features(bookings, as_of, rows["customer_id"]))
    feats = pd.concat(parts, ignore_index=True)

    # Recompute anything the spine already carries (e.g. recency_days) from the features.
    keep = [c for c in spine.columns if c not in feats.columns or c in ("customer_id", "as_of")]
    out = spine[keep].merge(feats, on=["customer_id", "as_of"], how="left", validate="one_to_one")
    out = out.merge(customers[["customer_id", "signup_date", "city", "acquisition_channel"]],
                    on="customer_id", how="left", validate="many_to_one")
    out["tenure_days"] = (out["as_of"] - pd.to_datetime(out["signup_date"])).dt.days
    return out.drop(columns="signup_date")


def snapshot_features(bookings, as_of, customer_ids):
    as_of = pd.Timestamp(as_of)
    past_all = bookings[bookings["day"] <= as_of]
    past = cd.add_gaps(past_all[past_all["customer_id"].isin(customer_ids)])
    past = _add_flags(past, as_of)

    snap = cd.customer_snapshot(past, as_of)
    service_gaps = cd.service_typical_gaps(past_all, as_of)  # norms use every customer
    g = past.groupby("customer_id")

    f = pd.DataFrame(index=snap.index)

    # --- Booking behaviour -------------------------------------------------------------
    f["recency_days"] = snap["recency_days"]
    f["days_since_first_booking"] = (as_of - g["day"].min()).dt.days
    f["n_bookings"] = snap["n_bookings"]
    f["n_bookings_30d"] = g["in_30d"].sum()
    f["n_bookings_90d"] = g["in_90d"].sum()
    # Trend: last 30 days vs the average 30 days in the 60 days before that.
    # Negative = slowing down, the "fading" pattern.
    prior_60d = f["n_bookings_90d"] - f["n_bookings_30d"]
    f["booking_trend_30d"] = f["n_bookings_30d"] - prior_60d / 2
    f["typical_gap_days"] = cd.typical_gap(snap, service_gaps)
    f["recency_vs_typical_gap"] = f["recency_days"] / f["typical_gap_days"]
    f["last_gap_vs_typical_gap"] = g["gap_days"].last() / f["typical_gap_days"]
    f["n_services_used"] = g["service_type"].nunique()
    f["main_service"] = snap["main_service"]

    # --- Experience (lifetime and last 90 days) -----------------------------------------
    f["customer_cancel_rate"] = g["is_customer_cancel"].mean()
    f["expert_cancel_rate"] = g["is_expert_cancel"].mean()
    f["expert_cancels_90d"] = g["expert_cancel_90d"].sum()
    f["avg_rating"] = g["rating"].mean()  # NaN if never rated
    f["rating_response_rate"] = g["rating"].count() / g["is_completed"].sum().replace(0, np.nan)
    f["low_ratings_90d"] = g["low_rating_90d"].sum()
    f["complaints_90d"] = g["complaint_90d"].sum()
    f["complaint_rate"] = g["had_complaint"].mean()
    f["avg_delay_minutes"] = g["expert_delay_minutes"].mean()
    f["late_arrivals_90d"] = g["late_90d"].sum()
    last = past.groupby("customer_id").tail(1).set_index("customer_id")
    f["last_booking_bad"] = last["is_bad"].astype(int)

    # --- Money ---------------------------------------------------------------------------
    completed = past[past["is_completed"]].groupby("customer_id")
    f["total_spend"] = completed["amount_paid"].sum().reindex(f.index, fill_value=0)
    f["avg_order_value"] = completed["amount_paid"].mean()
    f["coupon_rate"] = g["coupon_used"].mean()
    f["avg_discount_pct"] = (g["discount_amount"].sum() / g["list_price"].sum()).fillna(0)

    f = f.reset_index()
    f["as_of"] = as_of
    return f


def _add_flags(past, as_of):
    past = past.copy()
    days_ago = (as_of - past["day"]).dt.days
    past["in_30d"] = days_ago < 30
    past["in_90d"] = days_ago < 90
    past["is_completed"] = past["status"] == "completed"
    past["is_customer_cancel"] = past["status"] == "cancelled_by_customer"
    past["is_expert_cancel"] = past["status"] == "cancelled_by_expert"
    past["had_complaint"] = past["had_complaint"].astype(bool)
    is_low = past["rating"] <= LOW_RATING
    is_late = past["expert_delay_minutes"] > LATE_MINUTES
    past["is_bad"] = past["is_expert_cancel"] | is_low | past["had_complaint"] | is_late
    past["expert_cancel_90d"] = past["is_expert_cancel"] & past["in_90d"]
    past["low_rating_90d"] = is_low & past["in_90d"]
    past["complaint_90d"] = past["had_complaint"] & past["in_90d"]
    past["late_90d"] = is_late & past["in_90d"]
    return past
