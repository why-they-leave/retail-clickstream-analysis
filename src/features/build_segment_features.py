"""
고객 단위 집계 피처를 기반으로 segment assignment 입력 피처를 생성한다.

입력:
    data/processed/customer_features_all_customers.csv
    data/processed/customer_features_us_customers.csv
    data/interim/sessions_events_products.csv
    data/interim/orders_items_products.csv

출력:
    data/processed/segment_features_all_customers.csv
    data/processed/segment_features_us_customers.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CUSTOMER_FEATURE_PATHS = {
    "all_customers": PROCESSED_DIR / "customer_features_all_customers.csv",
    "us_customers": PROCESSED_DIR / "customer_features_us_customers.csv",
}

OUTPUT_PATHS = {
    "all_customers": PROCESSED_DIR / "segment_features_all_customers.csv",
    "us_customers": PROCESSED_DIR / "segment_features_us_customers.csv",
}

COLUMN_ORDER = [
    "customer_id",
    "page_view_count",
    "atc_rate",
    "order_count",
    "purchase_per_session",
    "total_spend_log",
    "recency_session_days",
    "recency_order_days",
    "view_purchase_category_match",
    "dominant_view_category_ratio",
    "dominant_purchase_category_ratio",
    "top_view_category",
    "top_purchase_category",
    "category_diversity_purchase",
    "session_count",
    "add_to_cart_count",
    "avg_order_value_log",
    "is_purchaser",
]


def _dominant_view_ratio(session_events: pd.DataFrame) -> pd.DataFrame:
    """고객별 최다 조회 카테고리 비율을 계산한다."""
    views = session_events[
        (session_events["event_type"] == "page_view") & session_events["category"].notna()
    ].copy()

    category_counts = (
        views.groupby(["customer_id", "category"]).size().reset_index(name="view_count")
    )
    total_counts = views.groupby("customer_id").size().reset_index(name="total_view_category_count")
    dominant_counts = (
        category_counts.sort_values(
            ["customer_id", "view_count", "category"], ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")[["customer_id", "view_count"]]
        .rename(columns={"view_count": "dominant_view_count"})
    )

    result = dominant_counts.merge(total_counts, on="customer_id", how="left")
    result["dominant_view_category_ratio"] = (
        result["dominant_view_count"] / result["total_view_category_count"]
    )
    return result[["customer_id", "dominant_view_category_ratio"]]


def _purchase_category_features(order_details: pd.DataFrame) -> pd.DataFrame:
    """고객별 구매 카테고리 다양성과 최다 구매 카테고리 비율을 계산한다."""
    purchases = order_details[order_details["category"].notna()].copy()
    if purchases.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "category_diversity_purchase",
                "dominant_purchase_category_ratio",
            ]
        )

    purchases["_purchase_weight"] = purchases["quantity"].fillna(0)

    category_quantities = (
        purchases.groupby(["customer_id", "category"])["_purchase_weight"]
        .sum()
        .reset_index(name="purchase_quantity")
    )
    total_quantities = (
        purchases.groupby("customer_id")["_purchase_weight"]
        .sum()
        .reset_index(name="total_purchase_quantity")
    )
    diversity = (
        purchases.groupby("customer_id")["category"]
        .nunique()
        .reset_index(name="category_diversity_purchase")
    )
    dominant_quantities = (
        category_quantities.sort_values(
            ["customer_id", "purchase_quantity", "category"], ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")[["customer_id", "purchase_quantity"]]
        .rename(columns={"purchase_quantity": "dominant_purchase_quantity"})
    )

    result = diversity.merge(dominant_quantities, on="customer_id", how="left").merge(
        total_quantities, on="customer_id", how="left"
    )
    result["dominant_purchase_category_ratio"] = np.where(
        result["total_purchase_quantity"] > 0,
        result["dominant_purchase_quantity"] / result["total_purchase_quantity"],
        np.nan,
    )
    return result[
        ["customer_id", "category_diversity_purchase", "dominant_purchase_category_ratio"]
    ]


def add_segment_features(
    customer_features: pd.DataFrame,
    session_events: pd.DataFrame,
    order_details: pd.DataFrame,
) -> pd.DataFrame:
    """확정된 segment feature schema에 맞게 파생 피처를 추가한다."""
    df = customer_features.copy()

    df["atc_rate"] = np.where(
        df["page_view_count"] > 0,
        df["add_to_cart_count"] / df["page_view_count"],
        np.nan,
    )
    df["purchase_per_session"] = np.where(
        df["session_count"] > 0,
        df["order_count"] / df["session_count"],
        np.nan,
    )
    df["total_spend_log"] = np.log1p(df["total_spend"])
    df["avg_order_value_log"] = np.log1p(df["avg_order_value"])
    df["is_purchaser"] = (df["order_count"] > 0).astype(int)

    view_notna = df["top_view_category"].notna()
    purchase_notna = df["top_purchase_category"].notna()
    category_same = (
        df["top_view_category"].astype("string").eq(df["top_purchase_category"].astype("string"))
    )
    df["view_purchase_category_match"] = (view_notna & purchase_notna & category_same).astype(int)

    df = df.merge(_dominant_view_ratio(session_events), on="customer_id", how="left").merge(
        _purchase_category_features(order_details), on="customer_id", how="left"
    )

    return df[COLUMN_ORDER]


def validate_segment_features(df: pd.DataFrame, label: str) -> None:
    """segment feature table의 기본 무결성을 검증한다."""
    errors = []
    if df["customer_id"].duplicated().any():
        errors.append("customer_id 중복 존재")
    for col in [
        "page_view_count",
        "order_count",
        "total_spend_log",
        "view_purchase_category_match",
    ]:
        if df[col].isna().any():
            errors.append(f"{col} 결측 존재")
    for col in ["atc_rate", "dominant_view_category_ratio", "dominant_purchase_category_ratio"]:
        values = df[col].dropna()
        if values.lt(0).any() or values.gt(1).any():
            errors.append(f"{col} 범위 오류")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"[{label}] segment feature 검증 실패:\n{details}")
    logger.info("[%s] segment feature 검증 통과 (rows=%s)", label, f"{len(df):,}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session_events = pd.read_csv(INTERIM_DIR / "sessions_events_products.csv")
    order_details = pd.read_csv(INTERIM_DIR / "orders_items_products.csv")

    for label, input_path in CUSTOMER_FEATURE_PATHS.items():
        customer_features = pd.read_csv(input_path)
        segment_features = add_segment_features(customer_features, session_events, order_details)
        validate_segment_features(segment_features, label)

        output_path = OUTPUT_PATHS[label]
        segment_features.to_csv(output_path, index=False)
        logger.info("[%s] 저장: %s", label, output_path)


if __name__ == "__main__":
    main()
