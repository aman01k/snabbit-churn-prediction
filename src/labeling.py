"""Turn the booking log into labeled training examples (one row per customer per snapshot).

A snapshot is a "pretend today" date. For each one we keep the customers who were still
active, and label whether they went on to book nothing during the next HORIZON_DAYS.
Features (next step) must only use bookings on or before the snapshot date.
"""

import pandas as pd

import churn_definition as cd
import config


def snapshot_dates(bookings):
    """Month-start dates that have enough history behind them and a full label window ahead."""
    first = bookings["day"].min() + pd.Timedelta(days=config.MIN_HISTORY_DAYS)
    last = bookings["day"].max() - pd.Timedelta(days=config.HORIZON_DAYS)
    return pd.date_range(first, last, freq="MS")


def assign_split(as_of):
    if as_of <= config.TRAIN_END:
        return "train"
    if as_of in config.VALID_DATES:
        return "valid"
    if as_of >= config.TEST_START:
        return "test"
    return "gap"  # buffer between periods, not used


def eligible_customers(bookings, as_of, excluded_services=None):
    """Customer snapshot at `as_of`, restricted to those the model scores.

    Only customers who are still active: already-churned customers need win-back, not
    retention, and including them would make the task trivially easy. Used both for
    training labels and for scoring, so the two populations always match.
    """
    if excluded_services is None:
        excluded_services = config.EXCLUDED_MAIN_SERVICES
    snap = cd.customer_snapshot(bookings, as_of)
    eligible = snap["recency_days"] < config.CHURN_INACTIVITY_DAYS
    eligible &= ~snap["main_service"].isin(excluded_services)
    return snap[eligible]


def label_snapshot(bookings, as_of, excluded_services=None):
    """Eligible customers at `as_of` with their label (1 = no booking in the next HORIZON_DAYS)."""
    snap = eligible_customers(bookings, as_of, excluded_services)
    returned = cd.returned_within(snap, bookings, as_of, config.HORIZON_DAYS)
    return pd.DataFrame({
        "customer_id": snap.index,
        "as_of": pd.Timestamp(as_of),
        "recency_days": snap["recency_days"].to_numpy(),
        "main_service": snap["main_service"].to_numpy(),
        "churned": (~returned).astype(int),
    })


def build_label_table(bookings, dates=None, excluded_services=None):
    dates = snapshot_dates(bookings) if dates is None else dates
    table = pd.concat([label_snapshot(bookings, d, excluded_services) for d in dates],
                      ignore_index=True)
    table["split"] = table["as_of"].map(assign_split)
    return table
