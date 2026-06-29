"""
고객 단위 집계 파이프라인 진입점.

실행 순서:
    1. preprocess_joins   — raw → 중간 조인 테이블 (data/interim/)
    2. build_customer_features — 중간 테이블 → 집계 피처 (data/processed/)

실행:
    python3 run_customer_pipeline.py
"""

from pathlib import Path

import pandas as pd
import preprocess_joins as pj
from build_customer_features import build_customer_features, validate

ROOT_DIR = Path(__file__).resolve().parents[2]  # 스크립트 파일 자체의 위치
RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 중간 조인 테이블 생성 ────────────────────────────────────────
    print("=== Step 1: 조인 테이블 생성 ===")
    pj.main()

    # ── Step 2: 중간 테이블 로드 ─────────────────────────────────────────────
    print("\n=== Step 2: 중간 테이블 로드 ===")
    session_events = pd.read_csv(INTERIM_DIR / "sessions_events_products.csv")
    order_details = pd.read_csv(INTERIM_DIR / "orders_items_products.csv")
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    print(f"  session_events: {len(session_events):,}행")
    print(f"  order_details:  {len(order_details):,}행")

    # ── Step 3: 전체 고객 기준 집계 ──────────────────────────────────────────
    print("\n=== Step 3: 전체 고객 피처 생성 ===")
    all_ids = customers["customer_id"]
    df_all = build_customer_features(all_ids, session_events, order_details)
    validate(df_all, expected_rows=len(customers), label="전체")
    df_all.to_csv(OUTPUT_DIR / "customer_features_all_customers.csv", index=False)
    print(f"  저장: {OUTPUT_DIR / 'customer_features_all_customers.csv'}")

    # ── Step 4: US 고객 기준 집계 ────────────────────────────────────────────
    print("\n=== Step 4: US-only 피처 생성 ===")
    us_ids = customers.loc[
        customers["country"] == "US", "customer_id"
    ]  # customers 테이블 기준으로 country 필터링
    df_us = build_customer_features(us_ids, session_events, order_details)
    validate(df_us, expected_rows=len(us_ids), label="US-only")  # US 고객 수 하드코딩 x
    df_us.to_csv(OUTPUT_DIR / "customer_features_us_customers.csv", index=False)
    print(f"  저장: {OUTPUT_DIR / 'customer_features_us_customers.csv'}")

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
