"""Model definitions, preprocessing and evaluation metrics for churn prediction."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "churned"

CATEGORICAL = ["main_service", "city", "acquisition_channel"]
NUMERIC = [
    "recency_days", "days_since_first_booking", "n_bookings", "n_bookings_30d", "n_bookings_90d",
    "booking_trend_30d", "typical_gap_days", "recency_vs_typical_gap", "last_gap_vs_typical_gap",
    "n_services_used", "customer_cancel_rate", "expert_cancel_rate", "expert_cancels_90d",
    "avg_rating", "rating_response_rate", "low_ratings_90d", "complaints_90d", "complaint_rate",
    "avg_delay_minutes", "late_arrivals_90d", "last_booking_bad", "total_spend",
    "avg_order_value", "coupon_rate", "avg_discount_pct", "tenure_days",
]
FEATURES = NUMERIC + CATEGORICAL

# Near-duplicates (notebook 03) that confuse a linear model's coefficients.
LINEAR_DROP = ["days_since_first_booking", "avg_discount_pct"]
# Long-tailed, non-negative features: log1p makes them friendlier for a linear model.
SKEWED = [
    "recency_days", "n_bookings", "n_bookings_30d", "n_bookings_90d", "typical_gap_days",
    "recency_vs_typical_gap", "last_gap_vs_typical_gap", "total_spend", "tenure_days",
    "avg_delay_minutes", "avg_order_value",
]


def linear_preprocessor():
    numeric = [c for c in NUMERIC if c not in LINEAR_DROP]
    skewed = [c for c in numeric if c in SKEWED]
    plain = [c for c in numeric if c not in SKEWED]
    impute_scale = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p), *impute_scale), skewed),
        ("plain", make_pipeline(*impute_scale), plain),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def tree_preprocessor():
    """Trees need no scaling and handle NaN themselves; categories become integer codes."""
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_models():
    """The candidates compared in notebook 04."""
    return {
        "logistic_regression": Pipeline([
            ("prep", linear_preprocessor()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "decision_tree": Pipeline([
            ("prep", tree_preprocessor()),
            ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", tree_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                             n_jobs=-1, random_state=SEED)),
        ]),
        "gradient_boosting": make_gradient_boosting(),
    }


def make_gradient_boosting(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100):
    return Pipeline([
        ("prep", tree_preprocessor()),
        ("model", HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
            categorical_features=CATEGORICAL, early_stopping=False, random_state=SEED)),
    ])


def evaluate(y, score, top_share=0.10):
    """Ranking, calibration and "contact the top X%" metrics for one set of predictions.

    `score` may be a probability or any risk score; brier is only meaningful for probabilities.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    k = int(round(len(y) * top_share))
    top = np.argsort(-score, kind="stable")[:k]
    precision_top = y[top].mean()
    is_prob = score.min() >= 0 and score.max() <= 1
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        f"precision_top{int(top_share * 100)}%": precision_top,
        f"recall_top{int(top_share * 100)}%": y[top].sum() / y.sum(),
        f"lift_top{int(top_share * 100)}%": precision_top / y.mean(),
        "brier": brier_score_loss(y, score) if is_prob else np.nan,
    }


def risk_segment(prob, high, medium):
    return pd.cut(prob, [-np.inf, medium, high, np.inf], right=False, labels=["Low", "Medium", "High"])
