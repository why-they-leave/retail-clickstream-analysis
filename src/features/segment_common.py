"""Segment assignment 공통 유틸리티."""

import pandas as pd

ZERO_FILL_CLUSTERING_COLS = [
    "page_view_count",
    "atc_rate",
    "order_count",
    "purchase_per_session",
    "total_spend_log",
    "view_purchase_category_match",
    "dominant_view_category_ratio",
    "dominant_purchase_category_ratio",
]

RECENCY_CLUSTERING_COLS = ["recency_session_days", "recency_order_days"]


def fill_clustering_features(
    df: pd.DataFrame,
    reference: pd.DataFrame,
    input_features: list[str],
) -> pd.DataFrame:
    """클러스터링 입력 피처 결측을 동일 규칙으로 대체한다."""
    missing_cols = sorted(set(input_features) - set(df.columns))
    if missing_cols:
        raise ValueError(f"segment feature table에 필요한 컬럼이 없습니다: {missing_cols}")

    result = df[input_features].copy()

    zero_fill_cols = [col for col in ZERO_FILL_CLUSTERING_COLS if col in result.columns]
    result[zero_fill_cols] = result[zero_fill_cols].fillna(0)

    for col in RECENCY_CLUSTERING_COLS:
        if col not in result.columns:
            continue
        if col not in reference.columns:
            raise ValueError(f"reference에 필요한 컬럼이 없습니다: {col}")
        max_value = reference[col].dropna().max()
        fill_value = max_value + 1 if pd.notna(max_value) else 0
        result[col] = result[col].fillna(fill_value)

    return result
