"""
make_als_mart.py
ALS 학습 및 평가를 위한 이벤트 레벨 데이터 생성
결과 csv 형식: [user_id, item_id, timestamp, score, event_type]

집계(groupby)는 als_model.py에서 train/test 분리 후 각각 수행
"""

import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_user_item_events():
    logger.info("데이터 로딩 중...")
    events = pd.read_csv("data/raw/events.csv")
    sessions = pd.read_csv("data/raw/sessions.csv")

    logger.info("데이터 전처리 시작...")
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    score_map = {"page_view": 1, "add_to_cart": 3, "checkout": 4, "purchase": 5}
    events["score"] = events["event_type"].map(score_map)

    unmapped_types = events.loc[events["score"].isnull(), "event_type"].unique()
    if len(unmapped_types) > 0:
        logger.warning(f"score_map에 없는 event_type 발견, score가 NaN으로 저장됨: {list(unmapped_types)}")

    # checkout/purchase의 product_id 복원용 세션별 장바구니 상품 목록
    # 검증 근거: checkout이 있는 전체 세션(44,909개)에서 add_to_cart qty 합 == cart_size 100% 일치.
    # 같은 세션에서 동일 상품이 qty만 다르게 두 번 기록되는 케이스(82건)가 있으므로
    # drop_duplicates 대신 qty를 합산하는 groupby 방식을 사용한다.
    cart_items = (
        events[events["event_type"] == "add_to_cart"]
        .groupby(["session_id", "product_id"])["qty"]
        .sum()
        .reset_index()
    )

    # product_id가 null인 이벤트는 checkout/purchase뿐이지만(DATA_CATALOG_raw.md 참고),
    # 암묵적 null 체크 대신 event_type을 명시해 의도를 드러낸다.
    cart_restore_types = ["checkout", "purchase"]
    item_events = events[~events["event_type"].isin(cart_restore_types)].copy()
    session_events = events[events["event_type"].isin(cart_restore_types)].copy()

    session_events = session_events.drop(columns=["product_id"])
    session_events_expanded = pd.merge(session_events, cart_items, on="session_id", how="inner")

    full_events = pd.concat([item_events, session_events_expanded], ignore_index=True)

    df_mart = pd.merge(
        full_events, sessions[["session_id", "customer_id", "country"]], on="session_id", how="left"
    )
    df_mart = df_mart.rename(columns={"customer_id": "user_id", "product_id": "item_id"})
    df_mart["user_id"] = df_mart["user_id"].astype(int)
    df_mart["item_id"] = df_mart["item_id"].astype(int)

    keep_cols = ["user_id", "item_id", "timestamp", "score", "event_type"]

    logger.info("1. 전체 유저 기준 이벤트 데이터 생성 중...")
    events_all = df_mart[keep_cols].copy()
    output_all = "data/processed/als_events.csv"
    events_all.to_csv(output_all, index=False)
    logger.info(f" -> 전체 이벤트 완료 ({len(events_all):,} 행)")

    logger.info("2. USA 유저 기준 이벤트 데이터 생성 중...")
    events_usa = df_mart[df_mart["country"] == "US"][keep_cols].copy()
    output_usa = "data/processed/als_events_us.csv"
    events_usa.to_csv(output_usa, index=False)
    logger.info(f" -> USA 이벤트 완료 ({len(events_usa):,} 행)")


if __name__ == "__main__":
    generate_user_item_events()
