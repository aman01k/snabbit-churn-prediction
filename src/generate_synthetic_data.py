"""Generate a synthetic Snabbit-like booking log for building the churn pipeline.

The output mimics what we expect from Snabbit's database: a customers table and a
bookings event log. It is NOT real data. Model results on it only show that the
pipeline works. See docs/synthetic_data_design.md for the behaviour rules.

Usage:
    python src/generate_synthetic_data.py --customers 20000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# service -> (typical days between bookings, list price in INR)
SERVICES = {
    "cleaning": (7, 299),
    "dishwashing": (4, 149),
    "laundry": (7, 199),
    "bathroom_cleaning": (21, 499),
    "deep_cleaning": (60, 1999),
}
PRIMARY_SERVICE_WEIGHTS = [0.40, 0.20, 0.20, 0.12, 0.08]

# segment -> (share of customers, booking-gap multiplier,
#             base quit probability per 30 days, bad-experience penalty multiplier)
# Quit risk is set per month, not per booking. Otherwise frequent bookers would face
# the risk more often and churn more than occasional ones. Loyal customers also
# forgive a bad visit more easily, so their penalties are scaled down.
SEGMENTS = {
    "loyal": (0.20, 1.0, 0.02, 0.3),
    "regular": (0.35, 1.5, 0.06, 0.6),
    "occasional": (0.30, 3.0, 0.10, 1.0),
    "trial": (0.15, 2.0, 0.50, 1.0),
}

CITIES = {
    "Mumbai": ["Andheri", "Bandra", "Powai", "Thane", "Malad"],
    "Bengaluru": ["Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "Jayanagar"],
    "Gurugram": ["DLF Phase 1", "Sohna Road", "Golf Course Road", "Sector 56"],
}
CITY_WEIGHTS = [0.40, 0.35, 0.25]
CITY_MEAN_DELAY_MIN = {"Mumbai": 12, "Bengaluru": 9, "Gurugram": 15}

CHANNELS = ["organic", "referral", "paid_social", "search_ads"]
CHANNEL_WEIGHTS = [0.35, 0.20, 0.25, 0.20]

NEVER_BOOK_SHARE = 0.08
CUSTOMER_CANCEL_RATE = 0.06
EXPERT_CANCEL_RATE = 0.04
RATING_RESPONSE_RATE = 0.65

# Extra quit probability added after a bad experience
PENALTY_EXPERT_CANCEL = 0.15
PENALTY_LOW_RATING = 0.12
PENALTY_COMPLAINT = 0.08
PENALTY_LATE = 0.05
LATE_THRESHOLD_MIN = 30


def make_customers(rng, n, start, end):
    """One row per customer, plus hidden traits that drive their behaviour."""
    # Leave two weeks at the end so even the newest customers can book once.
    signup_window = (end - start).days - 14
    signup_dates = start + pd.to_timedelta(rng.integers(0, signup_window, n), unit="D")

    cities = rng.choice(list(CITIES), n, p=CITY_WEIGHTS)
    localities = [rng.choice(CITIES[c]) for c in cities]

    seg_names = list(SEGMENTS)
    seg_shares = [SEGMENTS[s][0] for s in seg_names]

    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "signup_date": signup_dates,
        "city": cities,
        "locality": localities,
        "acquisition_channel": rng.choice(CHANNELS, n, p=CHANNEL_WEIGHTS),
        # Hidden traits below are dropped before saving customers.csv.
        "segment": rng.choice(seg_names, n, p=seg_shares),
        "primary_service": rng.choice(list(SERVICES), n, p=PRIMARY_SERVICE_WEIGHTS),
        "quality_sensitivity": rng.normal(0, 0.4, n),  # shifts how they rate
        "coupon_affinity": rng.beta(1, 4, n),
        "preferred_hour": rng.choice([8, 9, 10, 11, 14, 16, 18], n),
        "never_books": rng.random(n) < NEVER_BOOK_SHARE,
    })


def simulate_bookings(rng, cust, end):
    """Walk forward in time for one customer, booking until they quit or data ends.

    Returns (list of booking dicts, churn_date or None).
    churn_date is the date of the last booking before the customer decided to leave.
    None means they were still active when the data ends (right-censored).
    """
    if cust.never_books:
        return [], None

    _, gap_mult, monthly_quit, penalty_mult = SEGMENTS[cust.segment]
    typical_gap = SERVICES[cust.primary_service][0] * gap_mult
    # Convert the monthly rate to a per-booking chance for this customer's rhythm.
    base_quit = 1 - (1 - monthly_quit) ** (typical_gap / 30)

    bookings = []
    day = cust.signup_date + pd.Timedelta(days=int(rng.exponential(3)))
    fade_remaining = None  # bookings left before a fading customer stops
    stretch = 1.0  # fading customers book with ever-longer gaps

    while day <= end:
        service = cust.primary_service if rng.random() < 0.8 else rng.choice(list(SERVICES))
        booking = make_booking(rng, cust, service, day)
        bookings.append(booking)

        if fade_remaining is None:
            if rng.random() < quit_probability(base_quit, penalty_mult, booking):
                if rng.random() < 0.5:
                    return bookings, day  # leaves immediately
                fade_remaining = int(rng.integers(1, 4))
                stretch = rng.uniform(1.5, 2.5)
        else:
            fade_remaining -= 1
            if fade_remaining == 0:
                return bookings, day
            stretch *= rng.uniform(1.2, 1.6)

        # Gamma gives realistic jitter around the typical gap (never negative).
        gap = rng.gamma(shape=4, scale=typical_gap * stretch / 4)
        day += pd.Timedelta(days=max(1, round(gap)))

    return bookings, None


def make_booking(rng, cust, service, day):
    list_price = SERVICES[service][1]
    coupon_used = rng.random() < cust.coupon_affinity
    discount = round(list_price * rng.uniform(0.10, 0.30)) if coupon_used else 0

    r = rng.random()
    if r < CUSTOMER_CANCEL_RATE:
        status = "cancelled_by_customer"
    elif r < CUSTOMER_CANCEL_RATE + EXPERT_CANCEL_RATE:
        status = "cancelled_by_expert"
    else:
        status = "completed"

    delay = rating = np.nan
    had_complaint = False
    if status == "completed":
        delay = round(rng.exponential(CITY_MEAN_DELAY_MIN[cust.city]))
        if rng.random() < RATING_RESPONSE_RATE:
            score = 4.4 - cust.quality_sensitivity - 0.03 * delay + rng.normal(0, 0.6)
            rating = int(np.clip(round(score), 1, 5))
        unhappy = (not np.isnan(rating) and rating <= 2) or delay > 45
        had_complaint = unhappy and rng.random() < 0.6
    elif status == "cancelled_by_expert":
        had_complaint = rng.random() < 0.3

    hour = int(np.clip(cust.preferred_hour + rng.integers(-1, 2), 7, 20))
    return {
        "customer_id": cust.customer_id,
        "booking_datetime": day + pd.Timedelta(hours=hour),
        "service_type": service,
        "status": status,
        "list_price": list_price,
        "discount_amount": discount,
        "amount_paid": list_price - discount if status == "completed" else 0,
        "coupon_used": coupon_used,
        "expert_delay_minutes": delay,
        "rating": rating,
        "had_complaint": had_complaint,
    }


def quit_probability(base, penalty_mult, booking):
    penalty = 0.0
    if booking["status"] == "cancelled_by_expert":
        penalty += PENALTY_EXPERT_CANCEL
    if booking["rating"] <= 2:  # NaN compares False, so unrated bookings add nothing
        penalty += PENALTY_LOW_RATING
    if booking["had_complaint"]:
        penalty += PENALTY_COMPLAINT
    if booking["expert_delay_minutes"] > LATE_THRESHOLD_MIN:
        penalty += PENALTY_LATE
    return min(base + penalty_mult * penalty, 0.95)


def generate(n_customers, start, end, seed):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    customers = make_customers(rng, n_customers, start, end)

    all_bookings, truth = [], []
    for cust in customers.itertuples(index=False):
        bookings, churn_date = simulate_bookings(rng, cust, end)
        all_bookings.extend(bookings)
        truth.append({
            "customer_id": cust.customer_id,
            "segment": cust.segment,
            "primary_service": cust.primary_service,
            "never_booked": cust.never_books,
            "true_churn_date": churn_date,
        })

    bookings = pd.DataFrame(all_bookings).sort_values("booking_datetime", ignore_index=True)
    bookings.insert(0, "booking_id", [f"B{i:08d}" for i in range(1, len(bookings) + 1)])

    public_cols = ["customer_id", "signup_date", "city", "locality", "acquisition_channel"]
    return customers[public_cols], bookings, pd.DataFrame(truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--customers", type=int, default=20000)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/synthetic")
    args = parser.parse_args()

    customers, bookings, truth = generate(args.customers, args.start, args.end, args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    customers.to_csv(out / "customers.csv", index=False)
    bookings.to_csv(out / "bookings.csv", index=False)
    truth.to_csv(out / "_ground_truth.csv", index=False)

    print(f"customers: {len(customers):,}  bookings: {len(bookings):,}  -> {out}/")


if __name__ == "__main__":
    main()
