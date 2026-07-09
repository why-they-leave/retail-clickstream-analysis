"""LightGCN_tri 추천 결과 평가 (Issue #30). als_evaluate.py와 동일한 지표/구조.

사전 계산된 추천 결과(PRED_MAIN_RECOMMEND_<graph_mode>.csv, #34에서 tri/bipartite
파일명 분리) 기반 — 모델 재추론 없이 rank 컬럼으로 K를 조정하며 평가 가능.
정답셋(test)은 tri-graph의 valid u2t(구매만, #29에서 이미 ALS와 동일 split_date로
고정해둔 것)를 원본 id로 디코딩해서 즉석에서 만든다(ALS처럼 학습 시점에 별도 CSV로
저장해두지 않음 — 학습과 무관하게 매번 같은 값이라 재계산 비용이 거의 없고,
run_lightgcn.py를 다시 안 돌려도 평가만 재실행 가능).

Usage (cwd: src/baselines/lgcn3/):
    python3 evaluate_lightgcn.py
    python3 evaluate_lightgcn.py --k 5 10 20 50
    python3 evaluate_lightgcn.py --graph-mode bipartite
"""

# pandas보다 먼저 임포트 — 이 macOS 환경의 임포트 순서 데드락 회피
# (run_lightgcn.py, tests/conftest.py와 동일한 이유).
import tensorflow  # noqa: F401, I001

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARAMS_PATH = Path(__file__).resolve().parents[3] / "configs" / "LightGCN" / "params.yaml"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN"
LOG_DIR = str(Path(__file__).resolve().parents[3] / "logs" / "LightGCN")
PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

EVAL_FILE = "eval_results.csv"


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluate_lightgcn_{timestamp}.log")

    logger = logging.getLogger("evaluate_lightgcn")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_valid_pairs_as_df(valid_data: dict, user_dec: dict, item_dec: dict) -> pd.DataFrame:
    """tri_graph_uidx2tidx_valid.json({uidx: [tidx, ...]})을 user_id/item_id 원본 id로 디코딩한다.

    ALS의 test_pairs(구매만 정답)와 동일한 기준 — #29에서 이미 valid=구매만으로 고정해둠.
    """
    records = []
    for uidx_str, tidx_list in valid_data.items():
        uidx = int(uidx_str)
        if uidx not in user_dec:
            continue
        for tidx in tidx_list:
            if tidx in item_dec:
                records.append({"user_id": user_dec[uidx], "item_id": item_dec[tidx]})
    return pd.DataFrame(records, columns=["user_id", "item_id"])


def build_ground_truth(test_df: pd.DataFrame) -> dict:
    return test_df.groupby("user_id")["item_id"].apply(set).to_dict()


def hit_rate_at_k(recommended: list, ground_truth: set) -> float:
    """추천 상위 K개 중 정답이 1개라도 있으면 1, 없으면 0"""
    return 1.0 if len(set(recommended) & ground_truth) > 0 else 0.0


def recall_at_k(recommended: list, ground_truth: set) -> float:
    """실제 소비 아이템 중 추천 K개 안에 포함된 비율"""
    if len(ground_truth) == 0:
        return 0.0
    return len(set(recommended) & ground_truth) / len(ground_truth)


def ndcg_at_k(recommended: list, ground_truth: set) -> float:
    """추천 순위를 고려한 품질 지표 — 정답이 상위에 있을수록 높은 점수"""
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, item in enumerate(recommended, start=1)
        if item in ground_truth
    )
    ideal_hits = min(len(ground_truth), len(recommended))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_at_k(recs_df: pd.DataFrame, ground_truth: dict, k: int) -> dict:
    """저장된 추천 결과에서 rank <= k 필터링 후 지표 계산 (모델 재추론 없음)."""
    user_recs = (
        recs_df[recs_df["rank"] <= k]
        .sort_values("rank")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    hr_list, recall_list, ndcg_list = [], [], []
    skipped = 0
    for user_id, true_items in ground_truth.items():
        if user_id not in user_recs:
            skipped += 1
            continue
        recommended = user_recs[user_id]
        hr_list.append(hit_rate_at_k(recommended, true_items))
        recall_list.append(recall_at_k(recommended, true_items))
        ndcg_list.append(ndcg_at_k(recommended, true_items))

    return {
        "k": k,
        "HR": round(float(np.mean(hr_list)), 4) if hr_list else 0.0,
        "Recall": round(float(np.mean(recall_list)), 4) if recall_list else 0.0,
        "NDCG": round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        "eval_users": len(hr_list),
        "skipped": skipped,
    }


def main():
    from save_recommendations import GRAPH_MODE_FILE_SUFFIX, resolve_rec_filename
    from src.utils.id_encoding import build_id_encoding

    parser = argparse.ArgumentParser(description="LightGCN_tri 추천 결과 평가")
    parser.add_argument("--k", nargs="+", type=int, default=None, help="예: --k 5 10 20")
    parser.add_argument(
        "--graph-mode",
        choices=list(GRAPH_MODE_FILE_SUFFIX.keys()),
        default="tri",
        help="평가할 추천 결과 파일 선택 — tri: 페르소나 결합(기본) / bipartite: 대조군 (#34)",
    )
    args = parser.parse_args()

    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params_cfg = yaml.safe_load(f)
    k_list = args.k if args.k is not None else params_cfg.get("eval_k_list", [5, 10, 20])

    logger = setup_logging(LOG_DIR)
    logger.info(f"===== LightGCN_tri 평가 시작 | K={k_list} =====")

    # 1. 추천 결과 로드
    rec_path = OUTPUT_DIR / resolve_rec_filename(args.graph_mode)
    recs_df = pd.read_csv(rec_path)
    logger.info(
        f"[로드] 추천 결과: {len(recs_df):,}개 레코드 (유저 {recs_df['user_id'].nunique():,}명)"
    )

    # 2. 정답셋 로드 (valid u2t 디코딩)
    customer_ids = pd.read_csv(
        Path(__file__).resolve().parents[3] / "data" / "raw" / "customers.csv"
    )["customer_id"]
    product_ids = pd.read_csv(
        Path(__file__).resolve().parents[3] / "data" / "raw" / "products.csv"
    )["product_id"]
    _, user_dec = build_id_encoding(customer_ids)
    _, item_dec = build_id_encoding(product_ids)

    with open(PROCESSED_DIR / "tri_graph_uidx2tidx_valid.json", "r", encoding="utf-8") as f:
        valid_data = json.load(f)
    test_df = load_valid_pairs_as_df(valid_data, user_dec, item_dec)
    ground_truth = build_ground_truth(test_df)
    logger.info(f"[정답셋] 평가 대상 유저 수: {len(ground_truth):,}명")

    # 3. K별 평가
    results = [evaluate_at_k(recs_df, ground_truth, k) for k in k_list]
    for r in results:
        logger.info(
            f"[K={r['k']:2d}] 평가 유저: {r['eval_users']:,}명 / 스킵: {r['skipped']:,}명 | "
            f"HR@{r['k']}={r['HR']:.4f}  Recall@{r['k']}={r['Recall']:.4f}  NDCG@{r['k']}={r['NDCG']:.4f}"
        )

    # 4. 저장 (파일명에 graph_mode 반영 — tri/bipartite 평가 결과가 서로 덮어쓰지 않게, #34)
    eval_filename = EVAL_FILE.replace(".csv", f"_{args.graph_mode}.csv")
    eval_path = OUTPUT_DIR / eval_filename
    df_results = pd.DataFrame(results)
    df_results.insert(0, "model_type", f"LightGCN-{args.graph_mode}")
    df_results.to_csv(eval_path, index=False)
    logger.info(f"[저장] {eval_path}")
    logger.info("===== LightGCN_tri 평가 완료 =====")


if __name__ == "__main__":
    main()
