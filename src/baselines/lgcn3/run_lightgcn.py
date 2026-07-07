"""LightGCN_tri 학습 + 추천 저장 CLI (Issue #30). ALS(als_model.py) 스타일.

지금은 tri 모드만 지원한다 — bipartite(#35)는 read_data.py가 아직
bipartite_graph_*.json을 읽지 않아 #34에서 연동 예정
(docs/LIGHTGCN_TRI_MODEL_DESIGN.md 참고).

이 파일 상단부(순수 함수)는 params.py/read_data.py 등 무거운(임포트 시점에
CLI 인자를 읽는) 모듈을 임포트하지 않는다 — main() 안에서만 임포트해서,
`from run_lightgcn import build_recommendation_df`처럼 순수 함수만 가져다
테스트할 때 params.py의 argparse가 pytest의 sys.argv를 잘못 읽는 걸 막는다.

Usage (cwd: src/baselines/lgcn3/):
    python3 run_lightgcn.py --epoch 300
    python3 run_lightgcn.py --epoch 2   # 스모크 테스트용
"""

import argparse  # noqa: I001 (아래 tensorflow/pandas 임포트 순서는 의도적 — 재정렬 금지)
import logging
import os
from datetime import datetime
from pathlib import Path

# pandas보다 먼저 임포트해야 한다 — 이 macOS 환경에서 pandas를 먼저 임포트한 뒤
# tensorflow(이 파일 아래에서 read_data.py를 통해 간접적으로 임포트됨)를 임포트하면
# 임포트 자체가 데드락에 빠지는 문제가 있다 (tests/conftest.py와 동일한 원인,
# 여기선 스크립트라 conftest.py가 안 먹혀서 직접 임포트 순서를 고정해야 함).
# ruff의 import 정렬 자동수정(isort)이 이 순서를 도로 알파벳순으로 바꿔서
# 데드락을 재현시키므로 I001을 의도적으로 무시한다 — 절대 --fix로 재정렬하지 말 것.
import tensorflow  # noqa: F401, I001

import numpy as np
import pandas as pd
import yaml

PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "LightGCN" / "params.yaml"
LOG_DIR = str(Path(__file__).resolve().parents[3] / "logs" / "LightGCN")

# LightGCN_tri 고정 — parse.py의 MODEL_list 인덱스 (docs/LIGHTGCN_TRI_MODEL_DESIGN.md 참고)
MODEL_INDEX = 11
DATASET_INDEX = 2  # MBA — read_data.py가 우리 데이터를 이 이름으로 읽음


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_lightgcn_{timestamp}.log")

    logger = logging.getLogger("run_lightgcn")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_params(params_path: Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_recommendation_df(
    top_items: np.ndarray, top_scores: np.ndarray, user_ids: list, item_dec: dict
) -> pd.DataFrame:
    """모델의 top_items/top_scores(정수 인덱스)를 원본 id 기준 추천 CSV 스키마로 바꾼다.

    save_recommendations.save_lightgcn_recommendations()가 요구하는
    [user_id, item_id, score, rank]를 만족한다. top_k(sorted=True) 출력이라
    각 유저 행 순서가 이미 점수 내림차순이므로, 그 순서 그대로 rank 1..k를 매긴다.
    """
    records = []
    for u_index, user_id in enumerate(user_ids):
        for rank, (item_idx, score) in enumerate(
            zip(top_items[u_index], top_scores[u_index]), start=1
        ):
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_dec[int(item_idx)],
                    "score": round(float(score), 6),
                    "rank": rank,
                }
            )
    return pd.DataFrame(records)


def generate_all_user_recommendations(
    sess, model, train_data, user_num, item_num, top_n, batch_size=500
):
    """전체 유저(샘플 아님)에 대해 top_n 추천을 배치로 뽑는다.

    test_model.py의 test_model_store()는 TEST_USER_BATCH명만 샘플링하고 score도
    안 뽑는다 — PRED_MAIN_RECOMMEND.csv(rec-system #4 스키마)는 전체 유저 +
    score가 필요해서 별도로 구현했다. 배치 처리는 test_model_store()와 동일한
    이유(user_num*item_num 행렬을 한 번에 못 만듦)로 필요하다.
    """
    score_min = -(10**5)
    all_top_items = np.zeros((user_num, top_n), dtype=int)
    all_top_scores = np.zeros((user_num, top_n), dtype=np.float32)

    for start in range(0, user_num, batch_size):
        end = min(start + batch_size, user_num)
        user_batch = list(range(start, end))
        items_in_train_data = np.zeros((end - start, item_num), dtype=np.float32)
        for u_index, user in enumerate(user_batch):
            for item in train_data[user]:
                items_in_train_data[u_index, item] = score_min
        top_items, top_scores = sess.run(
            [model.top_items, model.top_scores],
            feed_dict={
                model.users: user_batch,
                model.keep_prob: 1,
                model.items_in_train_data: items_in_train_data,
                model.top_k: top_n,
            },
        )
        all_top_items[start:end] = top_items
        all_top_scores[start:end] = top_scores

    return all_top_items, all_top_scores


def main():
    parser = argparse.ArgumentParser(description="LightGCN_tri 학습 + 추천 저장")
    parser.add_argument(
        "--graph-mode",
        choices=["tri"],
        default="tri",
        help="tri만 지원 (bipartite는 #34에서 연동 예정)",
    )
    parser.add_argument("--epoch", type=int, default=None, help="미지정 시 params.yaml 값 사용")
    args = parser.parse_args()

    params_cfg = load_params(PARAMS_PATH)
    epoch = args.epoch if args.epoch is not None else params_cfg["epoch"]
    top_n = params_cfg["top_n"]

    logger = setup_logging(LOG_DIR)
    logger.info(f"===== LightGCN_tri 학습 시작 | graph_mode={args.graph_mode}, epoch={epoch} =====")

    # params.py는 임포트 시점에 sys.argv를 읽으므로, 여기서 명시적으로 구성한다
    # (model=LightGCN_tri, dataset=MBA로 고정 — 이 스크립트의 존재 이유 자체가 이 조합).
    import sys

    sys.argv = [
        "run_lightgcn",
        "--model",
        str(MODEL_INDEX),
        "--dataset",
        str(DATASET_INDEX),
        "--epoch",
        str(epoch),
        "--lr",
        str(params_cfg["lr"]),
        "--lamda",
        str(params_cfg["lamda"]),
        "--layer",
        str(params_cfg["layer"]),
    ]
    import params
    import read_data
    import train_model
    from save_recommendations import save_lightgcn_recommendations

    from src.utils.id_encoding import build_id_encoding

    # 1. 데이터 로딩 (tri-graph JSON + sparse propagation matrix)
    data = read_data.read_all_data_tri(params.all_para, approximate=False)
    train_data, user_num, item_num = data[0], data[2], data[3]
    logger.info(f"[데이터] user_num={user_num:,}, item_num={item_num:,}, persona_num={data[4]}")

    # 2. 학습 — sess/model을 돌려받아 학습 직후 같은 그래프로 전체 유저 추천을 뽑는다
    #    (train_model.py는 #30에서 F1_max 외에 sess/model도 반환하도록 확장함)
    excel_path = str(Path(LOG_DIR) / "run_lightgcn_train_log.xlsx")
    F1_max, sess, model = train_model.train_model(params.all_para, data, excel_path, "")
    logger.info(f"[학습 완료] F1_max={F1_max}")

    # 3. 전체 유저 추천 생성 (샘플이 아니라 user_num 전체, score 포함)
    customer_ids_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "customers.csv"
    product_ids_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "products.csv"
    customer_ids = pd.read_csv(customer_ids_path)["customer_id"]
    product_ids = pd.read_csv(product_ids_path)["product_id"]
    _, user_dec = build_id_encoding(customer_ids)
    _, item_dec = build_id_encoding(product_ids)

    logger.info(f"[추천 생성] 전체 유저 {user_num:,}명, top_n={top_n}")
    top_items, top_scores = generate_all_user_recommendations(
        sess, model, train_data, user_num, item_num, top_n
    )

    # 4. CSV 스키마 변환 + 저장 (rec-system #4 스키마, save_recommendations.py 재사용)
    user_ids = [user_dec[uidx] for uidx in range(user_num)]
    df_rec = build_recommendation_df(top_items, top_scores, user_ids, item_dec)
    output_path = save_lightgcn_recommendations(df_rec)
    logger.info(f"[저장] {output_path} ({len(df_rec):,}개 레코드)")

    logger.info("===== LightGCN_tri 학습 완료 =====")


if __name__ == "__main__":
    main()
