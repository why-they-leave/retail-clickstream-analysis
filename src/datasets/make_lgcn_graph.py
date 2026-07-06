"""
LightGCN용 tri-graph 입력(JSON) 생성 (Issue #29).

src/baselines/lgcn3/read_data.py가 요구하는 형식으로 유저-상품(u2t),
유저-세그먼트(u2p), 상품-세그먼트(t2p) 그래프를 만든다.

- 데이터 누수 방지: #31의 train 전용 세그먼트(customer_segments_labeled_train_only.csv)를
  사용하고, ALS(configs/ALS/params.yaml)와 동일한 split_date로 train/valid를 나눈다.
- u2t(학습용): 어떤 event_type을 엣지로 볼지 하나로 확정하지 않고 파라미터로 남겨둔다
  (Issue #29 코멘트 참고 — #30에서 조합별로 실험).
- valid(평가용) u2t: ALS의 평가 기준과 동일하게 "구매만" 정답으로 고정한다 — 이건
  실험 대상이 아니라 공정한 비교를 위해 반드시 지켜야 하는 기준이다.
- t2p: train 기간 구매 데이터로 Lift를 계산해, threshold를 넘는 세그먼트 전부와
  다중 연결한다(eric 정정, Issue #28/#29 코멘트 참고). lift 값은 edge weight로 쓴다
  (dense2sparse.py/read_data.py에 가중치 지원 추가함).

출력 (data/processed/, read_data.py의 DIR과 일치):
    tri_graph_uidx2tidx_train.json   {uidx: [tidx, ...]}
    tri_graph_uidx2tidx_valid.json   {uidx: [tidx, ...]}            (구매만)
    tri_graph_tidx2pidx.json         {tidx: [[segment_id, lift], ...]}
    tri_graph_uidx2pidx.json         {uidx: [segment_id]}           (단일 라벨)
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.id_encoding import build_id_encoding

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ALS(configs/ALS/params.yaml)의 split_date와 반드시 동일해야 공정한 비교가 된다.
CUTOFF_DATE = pd.Timestamp("2025-08-01")

# u2t(학습용) 엣지에 포함할 event_type — 실험 파라미터, 기본값은 전부 포함
DEFAULT_U2T_EVENT_TYPES = ["page_view", "add_to_cart", "purchase"]

# t2p Lift 기준 (eric 정정 — threshold 이상 전부 다중 연결, lift는 edge weight)
LIFT_THRESHOLD = 1.15
MIN_PURCHASE_COUNT = 5

TRAIN_U2T_PATH = PROCESSED_DIR / "tri_graph_uidx2tidx_train.json"
VALID_U2T_PATH = PROCESSED_DIR / "tri_graph_uidx2tidx_valid.json"
T2P_PATH = PROCESSED_DIR / "tri_graph_tidx2pidx.json"
U2P_PATH = PROCESSED_DIR / "tri_graph_uidx2pidx.json"

SEGMENT_LABELED_PATH = PROCESSED_DIR / "customer_segments_labeled_train_only.csv"


def load_catalog() -> tuple[dict, dict, dict, dict]:
    """전체 고객/상품 카탈로그 기준으로 인덱스 인코딩 테이블을 만든다.

    train/valid 그래프가 같은 인덱스 체계를 공유해야 하므로, train 기간에
    등장하지 않는 고객/상품도 전부 포함해서 인코딩한다.
    """
    customer_ids = pd.read_csv(RAW_DIR / "customers.csv")["customer_id"]
    product_ids = pd.read_csv(RAW_DIR / "products.csv")["product_id"]

    user_enc, user_dec = build_id_encoding(customer_ids)
    item_enc, item_dec = build_id_encoding(product_ids)
    logger.info("[카탈로그] user_num=%s, item_num=%s", len(user_enc), len(item_enc))
    return user_enc, user_dec, item_enc, item_dec


def load_events_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    """세션 이벤트/주문 원본을 로드한다 (날짜 컬럼 파싱 포함)."""
    session_events = pd.read_csv(INTERIM_DIR / "sessions_events_products.csv")
    order_details = pd.read_csv(INTERIM_DIR / "orders_items_products.csv")
    session_events["start_time"] = pd.to_datetime(session_events["start_time"])
    order_details["order_time"] = pd.to_datetime(order_details["order_time"])
    return session_events, order_details


def _interaction_pairs(
    session_events: pd.DataFrame, order_details: pd.DataFrame, event_types: list[str]
) -> pd.DataFrame:
    """event_types에 해당하는 (customer_id, product_id) 쌍을 모은다."""
    frames = []
    if "purchase" in event_types:
        frames.append(order_details[["customer_id", "product_id"]])
    session_types = [t for t in event_types if t != "purchase"]
    if session_types:
        frames.append(
            session_events.loc[
                session_events["event_type"].isin(session_types), ["customer_id", "product_id"]
            ].dropna()
        )
    if not frames:
        return pd.DataFrame(columns=["customer_id", "product_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def build_u2t_mapping(pairs: pd.DataFrame, user_enc: dict, item_enc: dict) -> dict[int, list[int]]:
    """(customer_id, product_id) 쌍을 uidx -> [tidx, ...]로 인코딩한다.

    user_num이 이 딕셔너리 길이로 결정되므로(read_data.py), 전체 고객이
    빠짐없이 키로 존재해야 한다 (상호작용이 없으면 빈 리스트).
    """
    pairs = pairs.copy()
    pairs["uidx"] = pairs["customer_id"].map(user_enc)
    pairs["tidx"] = pairs["product_id"].map(item_enc)
    pairs = pairs.dropna(subset=["uidx", "tidx"])
    pairs["uidx"] = pairs["uidx"].astype(int)
    pairs["tidx"] = pairs["tidx"].astype(int)

    grouped = pairs.groupby("uidx")["tidx"].apply(lambda s: sorted(set(int(v) for v in s)))
    return {uidx: grouped.get(uidx, []) for uidx in range(len(user_enc))}


def build_u2p_mapping(user_enc: dict, segment_labeled: pd.DataFrame) -> dict[int, list[int]]:
    """customer_segments_labeled_train_only.csv -> uidx -> [segment_id] (단일 라벨)."""
    segments = segment_labeled.copy()
    segments["uidx"] = segments["customer_id"].map(user_enc)
    segments = segments.dropna(subset=["uidx"])
    segments["uidx"] = segments["uidx"].astype(int)
    return {int(row.uidx): [int(row.segment_id)] for row in segments.itertuples()}


def build_t2p_mapping(
    order_details_train: pd.DataFrame,
    item_enc: dict,
    segment_labeled: pd.DataFrame,
) -> dict[int, list[list[float]]]:
    """train 기간 구매로 Lift를 계산해 item -> [[segment_id, lift], ...] 다중 연결을 만든다.

    lift = (segment 내 그 상품 구매 비율) / (전체 대비 그 segment 비율)
    threshold(LIFT_THRESHOLD)를 넘는 segment 전부와 연결한다(eric 정정 — 단일 연결 아님).
    최소 구매 건수(MIN_PURCHASE_COUNT) 미만인 상품은 lift가 불안정해 계산에서 제외한다.
    """
    purchases = order_details_train.merge(
        segment_labeled[["customer_id", "segment_id"]], on="customer_id", how="inner"
    )

    item_total_count = purchases.groupby("product_id").size()
    valid_items = item_total_count[item_total_count >= MIN_PURCHASE_COUNT].index

    segment_share = segment_labeled["segment_id"].value_counts(normalize=True)

    item_segment_count = (
        purchases[purchases["product_id"].isin(valid_items)]
        .groupby(["product_id", "segment_id"])
        .size()
    )

    result: dict[int, list[list[float]]] = {}
    for product_id in valid_items:
        tidx = item_enc.get(product_id)
        if tidx is None:
            continue
        total = item_total_count.loc[product_id]
        connected = []
        for segment_id, count in item_segment_count.loc[product_id].items():
            lift = (count / total) / segment_share[segment_id]
            if lift > LIFT_THRESHOLD:
                connected.append([int(segment_id), round(float(lift), 4)])
        if connected:
            result[tidx] = connected

    return {tidx: result.get(tidx, []) for tidx in range(len(item_enc))}


def save_json(data: dict, path: Path) -> None:
    """{int: ...} 딕셔너리를 read_data.py가 읽는 {str(int): ...} JSON으로 저장한다."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in data.items()}, f, ensure_ascii=False)
    logger.info("[저장] %s (%s개 키)", path, len(data))


def main(event_types: list[str] = DEFAULT_U2T_EVENT_TYPES) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    user_enc, _, item_enc, _ = load_catalog()
    session_events, order_details = load_events_orders()
    segment_labeled = pd.read_csv(SEGMENT_LABELED_PATH)[["customer_id", "segment_id"]]

    session_train = session_events[session_events["start_time"] < CUTOFF_DATE]
    order_train = order_details[order_details["order_time"] < CUTOFF_DATE]

    # train u2t: 실험 파라미터(event_types) 적용
    train_pairs = _interaction_pairs(session_train, order_train, event_types)
    train_u2t = build_u2t_mapping(train_pairs, user_enc, item_enc)
    save_json(train_u2t, TRAIN_U2T_PATH)

    # valid u2t: ALS 평가 기준과 동일하게 "구매만" 정답으로 고정 (실험 대상 아님)
    order_valid = order_details[order_details["order_time"] >= CUTOFF_DATE]
    valid_u2t = build_u2t_mapping(order_valid[["customer_id", "product_id"]], user_enc, item_enc)
    save_json(valid_u2t, VALID_U2T_PATH)

    # u2p
    u2p = build_u2p_mapping(user_enc, segment_labeled)
    save_json(u2p, U2P_PATH)

    # t2p
    t2p = build_t2p_mapping(order_train, item_enc, segment_labeled)
    save_json(t2p, T2P_PATH)

    logger.info(
        "완료: user_num=%s, item_num=%s, u2t event_types=%s",
        len(user_enc),
        len(item_enc),
        event_types,
    )


if __name__ == "__main__":
    main()
