"""Building blocks for defining churn from a booking event log.

Everything here takes an `as_of` date and only looks at bookings on or before it,
so the same functions can later build training labels without leaking the future.

Expected booking columns (map real Snabbit columns onto these when data arrives):
    customer_id, booking_datetime, service_type
"""

import numpy as np
import pandas as pd


def load_bookings(path):
    """Read the bookings log and add `day` (booking date without time)."""
    bookings = pd.read_csv(path, parse_dates=["booking_datetime"])
    bookings["day"] = bookings["booking_datetime"].dt.normalize()
    return bookings.sort_values(["customer_id", "day"], ignore_index=True)


def add_gaps(bookings):
    """Days since the same customer's previous booking (NaN for their first)."""
    out = bookings.copy()
    out["gap_days"] = out.groupby("customer_id")["day"].diff().dt.days
    return out


def customer_snapshot(bookings, as_of):
    """One row per customer who booked on or before `as_of`, describing them at that date.

    Columns: last_booking, n_bookings, main_service, personal_gap, recency_days.
    personal_gap is the customer's median gap between bookings (NaN if < 3 bookings).
    """
    as_of = pd.Timestamp(as_of)
    past = add_gaps(bookings[bookings["day"] <= as_of])

    snap = past.groupby("customer_id").agg(
        last_booking=("day", "max"),
        n_bookings=("day", "size"),
        main_service=("service_type", lambda s: s.mode().iloc[0]),
        personal_gap=("gap_days", "median"),
    )
    # A median of one or two gaps is too noisy to trust.
    snap.loc[snap["n_bookings"] < 3, "personal_gap"] = np.nan
    snap["recency_days"] = (as_of - snap["last_booking"]).dt.days
    return snap


def service_typical_gaps(bookings, as_of):
    """Median gap between bookings for customers grouped by their main service, up to `as_of`.

    Grouped by the customer's main service (not each booking's service), because a
    deep-cleaning customer who also books weekly cleaning should not drag the
    deep-cleaning norm down to 7 days.
    """
    past = add_gaps(bookings[bookings["day"] <= pd.Timestamp(as_of)])
    main_service = customer_snapshot(bookings, as_of)["main_service"]
    gaps = past.dropna(subset=["gap_days"]).join(main_service, on="customer_id")
    return gaps.groupby("main_service")["gap_days"].median()


def typical_gap(snap, service_gaps):
    """Customer's own rhythm if we trust it, otherwise the norm for their main service."""
    return snap["personal_gap"].fillna(snap["main_service"].map(service_gaps))


def returned_within(snap, bookings, as_of, horizon_days):
    """Observed outcome: did the customer book again within `horizon_days` after `as_of`?"""
    as_of = pd.Timestamp(as_of)
    window = bookings[
        (bookings["day"] > as_of) & (bookings["day"] <= as_of + pd.Timedelta(days=horizon_days))
    ]
    return snap.index.isin(window["customer_id"])


def churned_fixed(snap, days):
    """Rule A: churned if no booking for at least `days` days."""
    return snap["recency_days"] >= days


def churned_relative(snap, service_gaps, multiple, floor_days):
    """Rule B: churned if silent for `multiple` x their typical gap, and at least `floor_days`."""
    gap = typical_gap(snap, service_gaps)
    return (snap["recency_days"] >= multiple * gap) & (snap["recency_days"] >= floor_days)
