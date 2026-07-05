"""
중간 조인 테이블을 기반으로 고객 단위 집계 피처를 생성한다.

핵심 설계: customer_ids를 외부에서 주입받아 동일한 코드로 전체/US-only 결과를 생성한다.

입력:
    data/interim/session_events.csv
    data/interim/order_details.csv

출력:
    data/processed/customer_features_{suffix}.csv
"""

from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("../../data/interim")
OUTPUT_DIR = Path("../../data/processed")
ANALYSIS_DATE = pd.Timestamp("2025-11-01")

COLUMN_ORDER = [
    "customer_id",
    "recency_session_days",
    "recency_order_days",
    "session_count",
    "page_view_count",
    "add_to_cart_count",
    "order_count",
    "total_spend",
    "avg_order_value",
    "top_view_category",
    "top_purchase_category",
]


def _session_features(
    customer_ids: pd.Series, session_events: pd.DataFrame, analysis_date: pd.Timestamp
) -> pd.DataFrame:
    """recency_session_days, session_count 집계."""
    df = session_events[session_events["customer_id"].isin(customer_ids)].copy()
    df["start_time"] = pd.to_datetime(df["start_time"])

    agg = (
        df.groupby("customer_id")
        .agg(last_session=("start_time", "max"), session_count=("session_id", "nunique"))
        .reset_index()
    )
    agg["recency_session_days"] = (analysis_date - agg["last_session"]).dt.days
    return agg[["customer_id", "recency_session_days", "session_count"]]


def _event_features(customer_ids: pd.Series, session_events: pd.DataFrame) -> pd.DataFrame:
    """page_view_count, add_to_cart_count, top_view_category 집계."""
    df = session_events[session_events["customer_id"].isin(customer_ids)]

    counts = (
        df[df["event_type"].isin(["page_view", "add_to_cart"])]
        .groupby(["customer_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"page_view": "page_view_count", "add_to_cart": "add_to_cart_count"})
    )
    for col in ["page_view_count", "add_to_cart_count"]:
        if col not in counts.columns:
            counts[col] = 0

    top_view = (
        df[(df["event_type"] == "page_view") & df["product_id"].notna()]
        .groupby(["customer_id", "category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_id", "cnt", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .rename(columns={"category": "top_view_category"})[["customer_id", "top_view_category"]]
    )

    return counts.merge(top_view, on="customer_id", how="left")


def _order_features(
    customer_ids: pd.Series, order_details: pd.DataFrame, analysis_date: pd.Timestamp
) -> pd.DataFrame:
    """recency_order_days, order_count, total_spend, avg_order_value, top_purchase_category 집계."""
    df = order_details[order_details["customer_id"].isin(customer_ids)].copy()
    df["order_time"] = pd.to_datetime(df["order_time"])

    order_level = df.drop_duplicates(subset=["customer_id", "order_id"])  # 중복 집계 제거

    order_agg = (
        order_level.groupby("customer_id")
        .agg(
            last_order=("order_time", "max"),
            order_count=("order_id", "nunique"),
            total_spend=("total_usd", "sum"),
        )
        .reset_index()
    )
    order_agg["recency_order_days"] = (analysis_date - order_agg["last_order"]).dt.days
    order_agg["avg_order_value"] = order_agg["total_spend"] / order_agg["order_count"]

    top_purchase = (
        df.groupby(["customer_id", "category"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["customer_id", "quantity", "category"], ascending=[True, False, True])
        .drop_duplicates("customer_id")
        .rename(columns={"category": "top_purchase_category"})[
            ["customer_id", "top_purchase_category"]
        ]
    )

    return order_agg.merge(top_purchase, on="customer_id", how="left")[
        [
            "customer_id",
            "recency_order_days",
            "order_count",
            "total_spend",
            "avg_order_value",
            "top_purchase_category",
        ]
    ]


def build_customer_features(
    customer_ids: pd.Series,
    session_events: pd.DataFrame,
    order_details: pd.DataFrame,
    analysis_date: pd.Timestamp = ANALYSIS_DATE,
) -> pd.DataFrame:
    """customer_ids 기준으로 고객 단위 집계 피처 테이블을 생성한다."""
    base = pd.DataFrame({"customer_id": customer_ids})

    result = (
        base.merge(
            _session_features(customer_ids, session_events, analysis_date),
            on="customer_id",
            how="left",
        )
        .merge(_event_features(customer_ids, session_events), on="customer_id", how="left")
        .merge(
            _order_features(customer_ids, order_details, analysis_date),
            on="customer_id",
            how="left",
        )
    )

    result["session_count"] = result["session_count"].fillna(0).astype(int)
    result["page_view_count"] = result["page_view_count"].fillna(0).astype(int)
    result["add_to_cart_count"] = result["add_to_cart_count"].fillna(0).astype(int)
    result["order_count"] = result["order_count"].fillna(0).astype(int)
    result["total_spend"] = result["total_spend"].fillna(0.0)

    return result[COLUMN_ORDER]


def validate(df: pd.DataFrame, expected_rows: int, label: str) -> None:
    """생성된 피처 테이블의 무결성을 검증한다."""
    errors = []

    if len(df) != expected_rows:
        errors.append(f"row count {len(df):,} != expected {expected_rows:,}")
    if df["customer_id"].duplicated().any():
        errors.append("customer_id 중복 존재")
    for col in [
        "session_count",
        "page_view_count",
        "add_to_cart_count",
        "order_count",
        "total_spend",
    ]:
        if (df[col] < 0).any():
            errors.append(f"{col} 음수 존재")
    for col in ["recency_session_days", "recency_order_days"]:
        if df[col].dropna().lt(0).any():
            errors.append(f"{col} 음수 존재")

    if errors:
        details = "\n".join(f"  -{e}" for e in errors)
        raise ValueError(f"[{label}] 검증 실패: \n{details}")  # 예외 전파 추가
    else:
        print(f"[{label}] 검증 통과  (rows={len(df):,})")
