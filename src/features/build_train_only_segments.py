"""
LightGCN 실험용 — ALS와 동일한 train 기간 데이터만으로 세그먼트를 재계산한다 (Issue #31).

기존 `customer_segments_labeled_all_customers.csv`(#26)는 전체 기간(~2025-11-01)
데이터로 계산돼 ALS의 test 기간(2025-08-01 이후) 정보가 이미 섞여 있다. LightGCN이
이 세그먼트를 그대로 쓰면 test 기간 정보가 학습에 새어 들어가는 데이터 누수가 된다.

기존 파이프라인(build_customer_features -> build_segment_features -> assign_segments
-> segment_naming)의 함수를 그대로 재사용하되, 입력 데이터를 CUTOFF_DATE 이전으로
필터링해서 호출한다. 프로덕션 산출물(전체 기간 기준)은 건드리지 않고 별도 파일(
`*_train_only.*`)로 저장한다.

입력:
    data/raw/customers.csv
    data/interim/sessions_events_products.csv
    data/interim/orders_items_products.csv

출력:
    data/processed/customer_features_train_only.csv
    data/processed/segment_features_train_only.csv
    data/processed/customer_segments_train_only.csv
    data/processed/segment_summary_train_only.csv
    data/processed/segment_personas_train_only.json
    data/processed/customer_segments_labeled_train_only.csv
"""

import json
import logging
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.features.assign_segments import (
    assign_segment_ids,
    build_segment_summary,
    fit_segment_model,
    load_segment_config,
    validate_assignments,
)
from src.features.build_customer_features import build_customer_features, validate
from src.features.build_segment_features import (
    _dominant_view_ratio,
    _purchase_category_features,
    add_segment_features,
    validate_segment_features,
)
from src.llm_connector.env import get_required_env
from src.persona.segment_naming import (
    label_segment,
    merge_personas_with_customers,
    validate_merged_segments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ALS(configs/ALS/params.yaml)의 split_date와 반드시 동일해야 공정한 비교가 된다.
CUTOFF_DATE = pd.Timestamp("2025-08-01")

CUSTOMER_FEATURES_OUTPUT = PROCESSED_DIR / "customer_features_train_only.csv"
SEGMENT_FEATURES_OUTPUT = PROCESSED_DIR / "segment_features_train_only.csv"
CUSTOMER_SEGMENTS_OUTPUT = PROCESSED_DIR / "customer_segments_train_only.csv"
SEGMENT_SUMMARY_OUTPUT = PROCESSED_DIR / "segment_summary_train_only.csv"
SEGMENT_PERSONAS_OUTPUT = PROCESSED_DIR / "segment_personas_train_only.json"
LABELED_OUTPUT = PROCESSED_DIR / "customer_segments_labeled_train_only.csv"

PRODUCTION_LABELED_PATH = PROCESSED_DIR / "customer_segments_labeled_all_customers.csv"


def load_train_only_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """CUTOFF_DATE 이전 데이터만 남긴 session/order 테이블을 반환한다."""
    session_events = pd.read_csv(INTERIM_DIR / "sessions_events_products.csv")
    order_details = pd.read_csv(INTERIM_DIR / "orders_items_products.csv")

    session_events["start_time"] = pd.to_datetime(session_events["start_time"])
    order_details["order_time"] = pd.to_datetime(order_details["order_time"])

    session_train = session_events[session_events["start_time"] < CUTOFF_DATE].copy()
    order_train = order_details[order_details["order_time"] < CUTOFF_DATE].copy()

    logger.info(
        "[필터] session_events: %s -> %s / order_details: %s -> %s (기준: %s)",
        f"{len(session_events):,}",
        f"{len(session_train):,}",
        f"{len(order_details):,}",
        f"{len(order_train):,}",
        CUTOFF_DATE.date(),
    )
    return session_train, order_train


def build_features(session_train: pd.DataFrame, order_train: pd.DataFrame) -> pd.DataFrame:
    """train 기간 데이터로 customer_features -> segment_features를 생성한다."""
    customer_ids = pd.read_csv(RAW_DIR / "customers.csv")["customer_id"]

    customer_features = build_customer_features(
        customer_ids, session_train, order_train, analysis_date=CUTOFF_DATE
    )
    validate(customer_features, expected_rows=len(customer_ids), label="train_only")
    customer_features.to_csv(CUSTOMER_FEATURES_OUTPUT, index=False)
    logger.info("[저장] %s", CUSTOMER_FEATURES_OUTPUT)

    dominant_view_ratio = _dominant_view_ratio(session_train)
    purchase_category_features = _purchase_category_features(order_train)
    segment_features = add_segment_features(
        customer_features, dominant_view_ratio, purchase_category_features
    )
    validate_segment_features(segment_features, "train_only")
    segment_features.to_csv(SEGMENT_FEATURES_OUTPUT, index=False)
    logger.info("[저장] %s", SEGMENT_FEATURES_OUTPUT)
    return segment_features


def assign_segments(segment_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train 기간 피처로 KMeans를 새로 학습해 segment_id/segment_summary를 만든다."""
    config = load_segment_config()
    pipeline = fit_segment_model(segment_features, config)
    customer_segments = assign_segment_ids(segment_features, pipeline, config["input_features"])
    validate_assignments(customer_segments, "train_only", config["n_clusters"])
    customer_segments.to_csv(CUSTOMER_SEGMENTS_OUTPUT, index=False)
    logger.info("[저장] %s", CUSTOMER_SEGMENTS_OUTPUT)

    segment_summary = build_segment_summary(customer_segments)
    segment_summary.to_csv(SEGMENT_SUMMARY_OUTPUT, index=False)
    logger.info("[저장] %s", SEGMENT_SUMMARY_OUTPUT)
    return customer_segments, segment_summary


def name_segments(segment_summary: pd.DataFrame) -> list[dict]:
    """LLM으로 각 세그먼트에 이름/설명을 붙인다 (segment_naming.py의 라벨러 재사용)."""
    client = OpenAI(
        api_key=get_required_env("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
        timeout=60,
    )
    results = [
        label_segment(client, row)
        for _, row in segment_summary.sort_values("segment_id").iterrows()
    ]
    with open(SEGMENT_PERSONAS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("[저장] %s", SEGMENT_PERSONAS_OUTPUT)
    return results


def compare_with_production(customer_segments: pd.DataFrame) -> None:
    """기존(전체 기간) 세그먼트 분포와 비교해서 로그로 남긴다."""
    if not PRODUCTION_LABELED_PATH.exists():
        logger.warning(
            "[비교] 프로덕션 산출물이 없어 비교를 건너뜁니다: %s", PRODUCTION_LABELED_PATH
        )
        return

    production = pd.read_csv(PRODUCTION_LABELED_PATH)[["customer_id", "segment_id"]]
    merged = customer_segments[["customer_id", "segment_id"]].merge(
        production, on="customer_id", suffixes=("_train_only", "_all"), how="inner"
    )
    match_rate = (merged["segment_id_train_only"] == merged["segment_id_all"]).mean()
    logger.info(
        "[비교] train_only 분포:\n%s",
        customer_segments["segment_id"].value_counts().sort_index(),
    )
    logger.info(
        "[비교] 프로덕션(전체 기간) 분포:\n%s",
        production["segment_id"].value_counts().sort_index(),
    )
    logger.info("[비교] segment_id 일치율(같은 customer 기준): %.1f%%", match_rate * 100)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session_train, order_train = load_train_only_events()
    segment_features = build_features(session_train, order_train)
    customer_segments, segment_summary = assign_segments(segment_features)
    compare_with_production(customer_segments)

    personas = name_segments(segment_summary)
    merged = merge_personas_with_customers(customer_segments, personas)
    validate_merged_segments(merged, customer_segments, personas)
    merged.to_csv(LABELED_OUTPUT, index=False)
    logger.info("[저장] %s", LABELED_OUTPUT)


if __name__ == "__main__":
    main()
