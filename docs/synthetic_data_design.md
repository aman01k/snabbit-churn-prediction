# Synthetic Data Design

**Purpose:** we have no Snabbit data yet, so we generate a realistic stand-in to build
and test the churn pipeline. Every model result on this data only shows that the
pipeline works. It says **nothing** about Snabbit's real customers.

## Why a raw event log, not a ready-made churn table

Snabbit's database will most likely store *what happened* (customers signed up,
bookings were made, cancelled, rated). It will not have a `churned = yes/no` column.
Turning events into churn labels is the hardest and most error-prone part of the
project, so our practice data starts from the same place.

## Tables

### `customers.csv`: one row per customer

| Column | Meaning |
|---|---|
| `customer_id` | Unique ID (`C000001`) |
| `signup_date` | Date the account was created |
| `city`, `locality` | Where the customer lives |
| `acquisition_channel` | `organic`, `referral`, `paid_social`, `search_ads` |

### `bookings.csv`: one row per booking (the event log)

| Column | Meaning |
|---|---|
| `booking_id` | Unique ID |
| `customer_id` | Who booked |
| `booking_datetime` | When the service was scheduled |
| `service_type` | `cleaning`, `dishwashing`, `laundry`, `bathroom_cleaning`, `deep_cleaning` |
| `status` | `completed`, `cancelled_by_customer`, `cancelled_by_expert` |
| `list_price`, `discount_amount`, `amount_paid` | INR. `amount_paid` is 0 for cancelled bookings |
| `coupon_used` | Whether a coupon was applied |
| `expert_delay_minutes` | How late the expert arrived (completed bookings only) |
| `rating` | 1–5, blank if the customer did not rate or the booking was cancelled |
| `had_complaint` | Whether the customer raised a complaint |

### `_ground_truth.csv`: the answer key (never a feature)

Stores each customer's hidden segment and the date they actually decided to leave.
It exists only so we can check whether our labeling logic and model recover the
truth. Real data will **not** have this. Using it as a model input would be leakage.

## Hidden behaviour rules

We decide these rules ourselves, so we can later check whether the model finds them.

1. **Customer segments** control how often people book and how easily they leave:

   | Segment | Share | Booking gap vs. service norm | Base chance of quitting per 30 days |
   |---|---:|---:|---:|
   | loyal | 20% | 1.0× | 2% |
   | regular | 35% | 1.5× | 6% |
   | occasional | 30% | 3.0× | 10% |
   | trial | 15% | 2.0× | 50% |

   The monthly rate is converted to a per-booking chance based on each customer's
   booking rhythm. If the risk were set per booking instead, frequent bookers would face
   it more often and end up churning more than occasional ones, which is backwards.

2. **Each service has its own rhythm.** Dishwashing is about every 4 days, cleaning and laundry
   weekly, bathroom cleaning every 3 weeks, deep cleaning every 2 months. *This is why
   one inactivity cutoff for everyone may be wrong.*
3. **Bad experiences raise the chance of quitting:** expert cancellation +15%, rating ≤ 2 +12%,
   complaint +8%, expert more than 30 min late +5%. Loyal customers forgive more easily:
   these penalties are scaled by 0.3× for loyal and 0.6× for regular customers.
4. **Some quitters fade instead of leaving at once:** half of them make 1–3 more bookings
   with growing gaps first. This is what the "booking frequency trend" feature should catch.
5. **About 8% of sign-ups never book.**
6. **Right-censoring:** the data stops on an end date. A customer whose last booking is
   close to that date may simply not be due yet. Labeling them as churned would be
   wrong, and that's the trap the churn-definition step must handle.

## Swapping in the real data later

Only the loading step should change. Downstream code will depend on these concepts,
not on these exact column names: customer ID, booking timestamp, booking status,
amount, rating, and service type. Map the real columns onto them in one place.
