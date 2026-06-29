'''
# als_evaluate.py
ALS 추천 모델 평가 코드
- 평가 지표: HR@K, NDCG@K, Recall@K
- 가중치(pkl)와 테스트 데이터(csv) 경로는 상단 경로 설정에서 변경
'''

import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# 경로 설정 (자주 바꾸는 경우 여기만 수정)
# ============================================================
MODEL_PATH = "src/baselines/ALS/weights/als_model.pkl"
# MODEL_PATH = "src/baselines/ALS/weights/als_model_us.pkl"
TEST_PATH  = "data/outputs/ALS/als_test.csv"
# TEST_PATH  = "data/outputs/ALS/als_test_us.csv"

# ============================================================
# 상수 설정
# ============================================================
K = 20   # 평가 기준 상위 K개


# 1. 가중치 및 테스트 데이터 로드
def load_artifacts(model_path: str, test_path: str):
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    model    = artifacts["model"]
    user_enc = artifacts["user_enc"]
    item_enc = artifacts["item_enc"]
    item_dec = artifacts["item_dec"]

    test_df = pd.read_csv(test_path, parse_dates=["timestamp"])

    print(f"[로드 완료] 모델: {model_path}")
    print(f"[로드 완료] 테스트 데이터: {len(test_df):,}개 레코드")
    return model, user_enc, item_enc, item_dec, test_df


# 2. 유저별 정답 아이템 집합 생성
def build_ground_truth(test_df: pd.DataFrame) -> dict:
    """
    test_df에서 유저별 실제 소비한 item_id 집합을 만든다.
    """
    ground_truth = (
        test_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )
    print(f"[정답셋] 평가 대상 유저 수: {len(ground_truth):,}명")
    return ground_truth


# 3. 지표 계산 함수
def hit_rate_at_k(recommended: list, ground_truth: set) -> float:
    """
    추천 상위 K개 중 정답이 1개라도 있으면 1, 없으면 0
    """
    return 1.0 if len(set(recommended) & ground_truth) > 0 else 0.0


def recall_at_k(recommended: list, ground_truth: set) -> float:
    """
    실제 소비 아이템 중 추천 K개 안에 포함된 비율
    """
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(recommended) & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(recommended: list, ground_truth: set) -> float:
    """
    추천 순위를 고려한 품질 지표.
    정답이 상위에 있을수록 높은 점수.
    """
    dcg = 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: 정답 아이템이 1~len(ground_truth) 순위에 있을 때의 최대값
    ideal_hits = min(len(ground_truth), len(recommended))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# 4. 전체 평가 실행
def evaluate(model, user_enc: dict, item_enc: dict, item_dec: dict,
             ground_truth: dict, k: int):
    """
    test 유저 전체에 대해 HR@K, Recall@K, NDCG@K 계산
    - train에 없는 유저는 스킵 (학습 시 필터링된 유저)
    - 학습 시 소비한 아이템은 추천에서 제외 (filter_already_liked_items=True)
    """
    import scipy.sparse as sparse

    # 학습된 user/item factor 행렬로 빈 interaction 행렬 구성 (추천 시 필요)
    n_users = len(user_enc)
    n_items = len(item_enc)
    empty_matrix = sparse.csr_matrix((n_users, n_items))

    hr_list     = []
    recall_list = []
    ndcg_list   = []
    skipped     = 0

    for user_id, true_items in ground_truth.items():
        if user_id not in user_enc:
            skipped += 1
            continue

        u_idx = user_enc[user_id]

        item_indices, _ = model.recommend(
            u_idx,
            empty_matrix[u_idx],
            N=k,
            filter_already_liked_items=False  # test 평가이므로 학습 소비 아이템도 포함
        )

        recommended = [item_dec[i] for i in item_indices]

        hr_list.append(hit_rate_at_k(recommended, true_items))
        recall_list.append(recall_at_k(recommended, true_items))
        ndcg_list.append(ndcg_at_k(recommended, true_items))

    print(f"\n[평가 완료] K={k} / 평가 유저: {len(hr_list):,}명 / 스킵(신규 유저): {skipped:,}명")
    print(f"  HR@{k}     : {np.mean(hr_list):.4f}")
    print(f"  Recall@{k} : {np.mean(recall_list):.4f}")
    print(f"  NDCG@{k}   : {np.mean(ndcg_list):.4f}")

    return {
        f"HR@{k}"     : round(np.mean(hr_list), 4),
        f"Recall@{k}" : round(np.mean(recall_list), 4),
        f"NDCG@{k}"   : round(np.mean(ndcg_list), 4),
        "eval_users"  : len(hr_list),
        "skipped"     : skipped,
    }


# 메인 실행
if __name__ == "__main__":
    # 1. 로드
    model, user_enc, item_enc, item_dec, test_df = load_artifacts(MODEL_PATH, TEST_PATH)

    # 2. 정답셋 생성
    ground_truth = build_ground_truth(test_df)

    # 3. 평가
    results = evaluate(model, user_enc, item_enc, item_dec, ground_truth, K)