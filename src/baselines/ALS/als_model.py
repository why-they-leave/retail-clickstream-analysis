'''
# als_model.py
ALS 기반 메인 화면 개인화 추천 모델(ALS 학습 코드)
- Train/Test Split (시간 기준)
- 희소행렬 변환
- ALS 학습
- 유저별 추천 상위 50개 추출 및 저장
'''

import os
import pickle

import implicit
import numpy as np
import pandas as pd
import scipy.sparse as sparse

# 경로 설정 
MART_PATH  = "data/processed/als_datamart.csv"
# MART_PATH  = "data/processed/als_datamart_us.csv"
OUTPUT_DIR = "data/outputs/ALS"
MODEL_DIR  = "src/baselines/ALS"

# 상수 설정 
SPLIT_DATE    = "2025-08-01"   # train / test 분리 기준일
COLD_THRESHOLD = 10            # Cold 유저 판단 기준 (로그 N개 미만)
TOP_N         = 50             # 유저별 추천 후보 수


# 1. 데이터 로드 
def load_mart(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[로드 완료] 총 {len(df):,}개 레코드")
    return df


# 2. Train / Test Split 
def split_data(df: pd.DataFrame, split_date: str):
    train_df = df[df["timestamp"] < split_date].copy()
    test_df  = df[df["timestamp"] >= split_date].copy()
    print(f"[Split] Train: {len(train_df):,}개 / Test: {len(test_df):,}개")
    return train_df, test_df


# 3. 유저 세그먼트 분리 - 추론 시 사용할 내용
def segment_users(train_df: pd.DataFrame, threshold: int):
    """
    train 데이터 기준으로 Heavy / Cold 유저 분리
    threshold 미만 → Cold 유저 (인기도 기반 fallback)
    threshold 이상 → Heavy 유저 (ALS 추천)
    """
    user_log_counts = train_df.groupby("user_id")["item_id"].count()
    heavy_users = user_log_counts[user_log_counts >= threshold].index.tolist()
    cold_users  = user_log_counts[user_log_counts <  threshold].index.tolist()
    print(f"[세그먼트] Heavy 유저: {len(heavy_users):,}명 / Cold 유저: {len(cold_users):,}명")
    return heavy_users, cold_users


# 4. 희소행렬 변환
def build_sparse_matrix(train_df: pd.DataFrame):
    """
    [user_id, item_id, total_score] → CSR 희소행렬
    user/item을 0-based index로 인코딩
    """
    user_ids = train_df["user_id"].unique()
    item_ids = train_df["item_id"].unique()

    user_enc = {u: i for i, u in enumerate(user_ids)}
    item_enc = {it: i for i, it in enumerate(item_ids)}
    user_dec = {i: u for u, i in user_enc.items()}
    item_dec = {i: it for it, i in item_enc.items()}

    rows = train_df["user_id"].map(user_enc)
    cols = train_df["item_id"].map(item_enc)

    matrix = sparse.csr_matrix(
        (train_df["total_score"].astype(float), (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )
    print(f"[희소행렬] shape: {matrix.shape} / nonzero: {matrix.nnz:,}")
    return matrix, user_enc, item_enc, user_dec, item_dec


# 5. ALS 학습
def train_als(matrix: sparse.csr_matrix):
    """
    implicit 라이브러리 ALS 학습
    - factors   : 유저/아이템 임베딩 차원. 높을수록 표현력↑, 과적합 위험↑
    - iterations: 반복 횟수(U 고정→V 풀기, V 고정→U 풀기를 몇 번 반복할지)
    - alpha     : confidence 스케일 파라미터 (점수 * alpha = confidence)
                  alpha가 너무 크면 높은 score의 (user, item) 쌍에 과하게 편향되어
                  추천 다양성이 떨어짐. total_score에 이미 가중치가 반영된 경우
                  alpha=1부터 시작하여 추천 결과 다양성을 보며 조정할 것.
    """
    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        iterations=20,
        alpha=1,          # total_score에 가중치가 반영되어 있으므로 1부터 시작
        random_state=42
    )
    model.fit(matrix)
    print("[ALS] 학습 완료")
    return model


# 6. Heavy 유저 추천 생성
def generate_heavy_recommendations(
    model,
    matrix: sparse.csr_matrix,
    heavy_users: list,
    user_enc: dict,
    item_dec: dict,
    top_n: int
) -> pd.DataFrame:
    """
    Heavy 유저에 대해 ALS 추천 상위 N개 추출
    이미 소비한 아이템은 제외 (filter_already_liked_items=True)
    """
    records = []
    for user_id in heavy_users:
        if user_id not in user_enc:
            continue
        u_idx = user_enc[user_id]
        item_ids_rec, scores = model.recommend(
            u_idx,
            matrix[u_idx],
            N=top_n,
            filter_already_liked_items=True
        )
        for rank, (item_idx, score) in enumerate(zip(item_ids_rec, scores), start=1):
            records.append({
                "user_id"  : user_id,
                "item_id"  : item_dec[item_idx],
                "score"    : round(float(score), 6),
                "rank"     : rank,
                "user_type": "heavy"
            })

    df_rec = pd.DataFrame(records)
    print(f"[Heavy 추천] {len(heavy_users):,}명 → {len(df_rec):,}개 레코드 생성")
    return df_rec


# 7. Cold 유저 인기도 기반 Fallback 추천
def generate_cold_recommendations(
    train_df: pd.DataFrame,
    cold_users: list,
    top_n: int
) -> pd.DataFrame:
    """
    Cold 유저에게는 전체 train 기준 인기 상품(total_score 합산) 상위 N개 추천
    """
    popular_items = (
        train_df.groupby("item_id")["total_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    popular_items.columns = ["item_id", "score"]
    popular_items["rank"] = range(1, len(popular_items) + 1)

    records = []
    for user_id in cold_users:
        temp = popular_items.copy()
        temp["user_id"]   = user_id
        temp["user_type"] = "cold"
        records.append(temp)

    df_cold = pd.concat(records, ignore_index=True)
    print(f"[Cold 추천] {len(cold_users):,}명 → {len(df_cold):,}개 레코드 생성")
    return df_cold


# 8. 결과 저장
def save_outputs(
    df_rec: pd.DataFrame,
    model,
    user_enc: dict,
    item_enc: dict,
    user_dec: dict,
    item_dec: dict,
    output_dir: str,
    model_dir: str
):
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # 추천 결과 저장
    rec_path = os.path.join(output_dir, "PRED_MAIN_RECOMMEND.csv")
    # rec_path = os.path.join(output_dir, "PRED_MAIN_RECOMMEND_US.csv")
    df_rec.to_csv(rec_path, index=False)
    print(f"[저장] 추천 결과 → {rec_path}")

    # 모델 및 인코더 저장 (추후 추론 시 재사용)
    model_path = os.path.join(weights_dir, "als_model.pkl")
    # model_path = os.path.join(weights_dir, "als_model_us.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model"   : model,
            "user_enc": user_enc,
            "item_enc": item_enc,
            "user_dec": user_dec,
            "item_dec": item_dec,
        }, f)
    print(f"[저장] 모델 → {model_path}")


# 메인 실행
if __name__ == "__main__":
    # 1. 로드
    mart = load_mart(MART_PATH)

    # 2. Split
    train_df, test_df = split_data(mart, SPLIT_DATE)

    # 3. 유저 세그먼트
    heavy_users, cold_users = segment_users(train_df, COLD_THRESHOLD)

    # 4. 희소행렬
    matrix, user_enc, item_enc, user_dec, item_dec = build_sparse_matrix(train_df)

    # 5. ALS 학습
    model = train_als(matrix)

    # 6. Heavy 유저 추천
    df_heavy = generate_heavy_recommendations(
        model, matrix, heavy_users, user_enc, item_dec, TOP_N
    )

    # 7. Cold 유저 추천
    df_cold = generate_cold_recommendations(train_df, cold_users, TOP_N)

    # 8. 합치기
    df_all = pd.concat([df_heavy, df_cold], ignore_index=True)
    print(f"\n[최종] 전체 추천 레코드: {len(df_all):,}개")

    # 9. 저장
    save_outputs(df_all, model, user_enc, item_enc, user_dec, item_dec, OUTPUT_DIR, MODEL_DIR)

    # 10. test_df 저장 (평가 시 사용)
    # train에 없는 신규 유저/아이템 비율 확인 후 필터링
    new_users = test_df[~test_df["user_id"].isin(user_enc)]
    new_items = test_df[~test_df["item_id"].isin(item_enc)]
    print(f"[신규 유저] test 내 train 미등장 유저 레코드: {len(new_users):,}개 ({len(new_users)/len(test_df)*100:.1f}%)")
    print(f"[신규 아이템] test 내 train 미등장 아이템 레코드: {len(new_items):,}개 ({len(new_items)/len(test_df)*100:.1f}%)")

    test_df_filtered = test_df[
        test_df["user_id"].isin(user_enc) &
        test_df["item_id"].isin(item_enc)
    ]
    print(f"[필터링 후] test 레코드: {len(test_df_filtered):,}개")

    test_path = os.path.join(OUTPUT_DIR, "als_test.csv")
    # test_path = os.path.join(OUTPUT_DIR, "als_test_us.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_df_filtered.to_csv(test_path, index=False)
    print(f"[저장] 테스트 데이터 → {test_path}")