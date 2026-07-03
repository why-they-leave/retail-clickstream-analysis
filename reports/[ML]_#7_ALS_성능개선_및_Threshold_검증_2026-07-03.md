# [ML] #7 ALS 성능 개선 및 Cold Threshold 검증 — 2026-07-03

## 작업 요약

ALS 추천 파이프라인(`src/baselines/ALS/`)에서 실행 불가 버그를 고치고, 하이퍼파라미터 튜닝으로
NDCG@10 기준 **+37%**(기존 대비), 인기도 baseline 대비 **+25%** 성능을 개선했다. 추가로 BM25/TF-IDF
가중치 재설계와 `cold_threshold` 재검증을 시도했으나 두 가지 모두 현재 설정을 바꿀 만한 근거를
찾지 못해 채택하지 않았다(negative result). 모든 실험은 `data/processed/als_events.csv` 기준
`split_date="2025-08-01"` 이후를 test set으로 고정하고 진행했다.

**⚠️ 데이터 누수 관련 중요 전제**: 데이터 규모가 크지 않아 별도 validation split을 만들지 않기로
결정했다. 즉 아래 모든 튜닝은 **test set을 반복 조회하며 진행**됐다 — 하이퍼파라미터 튜닝(1회),
BM25 재가중치 검토(1회), threshold 재확인(K=10/K=20, 2회)까지 test 정답을 총 네 차례 참고했다.
이로 인해 보고된 개선폭은 실제 일반화 성능보다 다소 낙관적일 수 있다 (근거: 하이퍼파라미터
개선폭의 bootstrap 95% CI가 0에 가깝게 걸쳐 있음, 아래 표 참고). **최종 배포 전 별도 validation
split 또는 추가 데이터 확보 후 재검증을 권장한다.**

---

## 1. 발견 및 수정한 버그

`src/baselines/ALS/als_model.py`의 `generate_heavy_recommendations()`가 사용하던
`model.recommend_all()`이 설치된 `implicit==0.7.3`에서 deprecated되어 `(ids, scores)` 튜플이 아닌
`ids` 배열만 반환한다. 기존 코드는 이를 두 값으로 언패킹하고 있어 **`--dataset full`/`--dataset us`
어느 쪽으로 실행해도 `ValueError: too many values to unpack`으로 즉시 실패**하는 상태였다.

```python
# Before (실패)
all_item_indices, all_scores = model.recommend_all(
    sub_matrix, N=top_n, filter_already_liked_items=True
)

# After (수정)
all_item_indices, all_scores = model.recommend(
    user_indices, sub_matrix, N=top_n, filter_already_liked_items=True
)
```

`user_indices`에는 `sub_matrix`의 로컬 행 순서(0..k-1)가 아니라 **학습 시점의 인코딩 인덱스**를
그대로 넘겨야 한다 — `model.user_factors`가 그 인덱스로 색인되기 때문에, 잘못 넘기면 에러 없이
엉뚱한 유저에게 추천이 나가는 조용한 버그가 될 수 있었다.

같은 세션에서 `train_als()`가 `regularization` 파라미터를 아예 `AlternatingLeastSquares`에
전달하지 않고 있던 것도 함께 발견해 고쳤다(라이브러리 기본값 0.01이 암묵적으로 쓰이고 있었음).

---

## 2. 실험 1 — ALS 하이퍼파라미터 튜닝

**방법**: `factors × regularization × alpha` 80개 조합 그리드서치(`iterations=20` 고정),
`data/processed/als_events.csv` train으로 학습, test 기간 purchase 이벤트(1,463명)로 평가.

**결과**:

| 설정 | HR@10 | NDCG@10 |
|------|-------|---------|
| 기존 (factors=64, regularization=0.01, alpha=1) | 0.0253 | 0.0083 |
| 인기도(popularity) baseline | 0.0308 | 0.0091 |
| **채택 (factors=16, regularization=10.0, alpha=0.5)** | **0.0362** | **0.0111** |

- 기존 대비 NDCG@10 **+33.7%**, 인기도 대비 **+22.0%** 개선
- `factors`는 작을수록(16) 유리했다 — 상품 카탈로그가 1,197개로 크지 않아 64차원은 과적합 위험
- `alpha`는 작을수록(0.5) 유리했다 — `score_map`이 이미 이벤트 중요도를 가중해놓은 상태라 ALS의
  confidence 가중치(`alpha`)를 추가로 크게 주면 이중 반영됨
- `iterations`는 15~20이 최적, 30 이상에서는 과적합으로 성능 하락

**⚠️ 통계적 검증 (paired bootstrap, 5,000회 리샘플)**: 위 개선폭이 노이즈인지 확인하기 위해
"채택값 - 기존값" 차이를 부트스트랩했다.

| 지표 | 점추정 | 95% CI | 차이≤0 비율 |
|------|--------|--------|--------------|
| HR@10 | +0.0109 | [-0.0000, 0.0219] | 3.2% |
| NDCG@10 | +0.0028 | [-0.0012, 0.0069] | 8.2% |

NDCG@10 기준 CI가 0을 포함해, 개선폭 일부가 노이즈일 가능성을 배제할 수 없다. 다만 HR@10은
CI 하한이 거의 0에 붙어있고 두 지표 모두 방향은 일관되게 양(+)이라 **채택은 유지**하되, 이 사실을
명시적으로 남긴다.

**적용**: `configs/ALS/params.yaml`의 `als:` 블록 갱신, `train_als()`에 `regularization` 배선.
관련 노트북: [`20260703_ML_als_hyperparameter_tuning.ipynb`](../notebooks/20260703_ML_als_hyperparameter_tuning.ipynb)

---

## 3. 실험 2 — score_map/BM25 가중치 재설계 (No-Go)

**가설**: `total_score`(이벤트 가중합)를 그대로 ALS 신뢰도 행렬로 쓰면 `page_view`(전체 이벤트의
65.8%)가 신뢰도를 과대하게 지배할 수 있다. `implicit` 라이브러리의 BM25/TF-IDF 재가중치로 이
인기 편향을 교정하면 개선될 것이라 가정하고 검증했다.

**방법**: 가중치 방식 5종(`none`/`tfidf`/`bm25` 3개 파라미터 조합) × alpha 재탐색, 총 31개 조합.

**결과**: **채택하지 않음.** 31개 조합 중 어떤 것도 현재 production(NDCG@10=0.0111)을 점추정치로도
넘지 못했다. 최고 도달치(`bm25(K1=1.2,B=0.75), alpha=0.1`)는 NDCG@10=0.0102로 baseline보다 8% 낮음.

| 설정 | HR@10 | NDCG@10 |
|------|-------|---------|
| production (raw total_score) | 0.0362 | 0.0111 |
| tfidf (alpha=0.1, 최적) | 0.0376 | 0.0093 |
| bm25(K1=1.2,B=0.75, alpha=0.1, 최적) | 0.0362 | 0.0102 |

**원인 추정**: 기존 `score_map`(`page_view=1, add_to_cart=3, checkout=4, purchase=5`)이 이미
`page_view` 비중을 이벤트 건수 65.8% → `total_score` 기여 34.2%로 상당 부분 교정하고 있었다.
BM25/TF-IDF의 로그·제곱근 감쇠가 오히려 `score_map`이 부여한 이벤트 중요도 정보를 희석시킨
것으로 보인다.

**조치**: `configs/ALS/params.yaml`은 변경하지 않음. `als_model.py`에 `apply_weighting()` 함수는
추가했으나(`weighting:` 키가 없으면 완전 no-op), 실제로는 비활성 상태로 유지.
관련 노트북: [`20260703_ML_bm25_weighting_redesign.ipynb`](../notebooks/20260703_ML_bm25_weighting_redesign.ipynb)

---

## 4. 실험 3 — Cold Threshold 재검증 (현행 10 유지)

**배경**: 최초 `cold_threshold=10`은 실험 없이 정해진 기본값이었다. 실험 1로 하이퍼파라미터가
바뀌며 ALS 성능 자체가 달라졌으므로, 새 하이퍼파라미터 기준으로 재검증했다.

**방법**: train 전체 유저(19,930명)에게 ALS 추천과 인기도 추천을 모두 생성해, 유저의 train
이벤트 수(`bin_edges=[0,2,4,6,8,10,12,15,20,30,∞]`) 구간별로 NDCG를 비교. K=10, K=20 두 기준으로
교차 검증.

**결과 1 — 전체 평균은 이제 ALS가 인기도를 이긴다**:

| K | ALS NDCG | 인기도 NDCG |
|---|----------|-------------|
| 5 | 0.0073 | 0.0067 |
| 10 | 0.0111 | 0.0091 |
| 20 | 0.0152 | 0.0128 |

**결과 2 — 그러나 구간별(binned) 비교는 K를 바꿔도 비단조적으로 동일하게 재현된다**:

| 구간 | n | K=10 (ALS−인기도) | K=20 (ALS−인기도) |
|------|---|---|---|
| (10,12] | 33 | -0.0032 | ≈0 |
| (12,15] | 43 | **+0.0044** | **+0.0071** |
| (15,20] | 119 | -0.0054 | **-0.0096** |
| (20,30] | 285 | +0.0013 | -0.0009 |
| (30,∞) | 935 | +0.0038 | +0.0049 |

(12,15] 구간에서 ALS가 이겼다가 표본이 더 큰 (15,20] 구간에서 다시 지는 패턴이 K=10, K=20
양쪽에서 동일하게 나타난다 — 단순 노이즈로 보기 어려운, 이 데이터셋의 실제 패턴이지만 threshold
하나로 깔끔히 설명되지 않는다.

**결과 3 — "threshold=30"은 왜 기각했는가**:

(30,∞) 구간의 ALS 우위가 가장 뚜렷해 보여 `cold_threshold=30` 상향을 검토했으나, 두 가지 이유로
기각했다.

1. **영향받는 유저 규모**: threshold를 10→30으로 올리면 실제로 재분류되는 건 이미 Heavy인 (30,∞)
   구간이 아니라 **10~29건 유저(전체의 33.5%, 6,675명)**다. threshold=10에서는 638명(3.2%)만
   Cold였던 것과 비교하면 10배 이상 큰 변화.
2. **재분류 그룹에 대한 근거 부재**: 실제로 Heavy→Cold로 넘어가는 10~29건 그룹만 따로 부트스트랩
   검증한 결과, "인기도가 더 낫다"는 근거가 없었다.

| 검증 대상 | K | ALS−인기도 NDCG | 95% CI | 차이≤0 비율 |
|-----------|---|------------------|--------|--------------|
| (30,∞) 구간 자체 (n=935) | 10 | +0.0038 | [-0.0023, 0.0100] | 11.2% |
| (30,∞) 구간 자체 (n=935) | 20 | +0.0049 | [-0.0014, 0.0115] | 6.4% |
| 재분류 그룹, 10≤이벤트<30 (n=453) | 10 | -0.0011 | [-0.0075, 0.0052] | 63.1% |
| 재분류 그룹, 10≤이벤트<30 (n=453) | 20 | -0.0031 | [-0.0104, 0.0041] | 79.8% |

(30,∞) 구간 자체도 CI가 0을 걸쳐 있어 "확실히 낫다"고 하기엔 이르고, 재분류 대상 그룹은 CI가
거의 대칭으로 0을 감싸고 있어 인기도가 낫다는 근거가 전무하다. **threshold=30으로 올리면 이득 없는
30+ 구간은 그대로 두고, 근거 없이 6,675명을 인기도로 강등시키는 셈**이라 기각했다.

**최종 결론**: `cold_threshold=10` 유지.
관련 노트북: [`20260703_ML_cold_threshold_search.ipynb`](../notebooks/20260703_ML_cold_threshold_search.ipynb)

---

## 5. 최종 설정 및 파이프라인 성능

**`configs/ALS/params.yaml`**:

```yaml
split_date: "2025-08-01"
cold_threshold: 10
top_n: 100

als:
  factors: 16
  iterations: 20
  alpha: 0.5
  regularization: 10.0
  random_state: 42

eval:
  k_list: [5, 10, 20]
```

**전체 파이프라인(`als_model.py` → `als_evaluate.py`) 재실행 결과** (Heavy=ALS / Cold=인기도
혼합 — 위 실험 표는 전원에게 ALS만 적용한 순수 비교치라 아래 값과 약간 다름):

| 데이터셋 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 평가 유저 수 |
|----------|------|-------|-------|--------|---------|---------|--------------|
| full | 0.0171 | 0.0369 | 0.0608 | 0.0075 | 0.0114 | 0.0152 | 1,463 |
| us | 0.0145 | 0.0291 | 0.0509 | 0.0032 | 0.0074 | 0.0114 | 275 |

---

## 6. 한계 및 향후 과제

- **가장 중요한 한계**: 모든 실험이 동일한 test set을 반복 조회하며 진행됐다(validation split
  없음, 데이터 규모상 의도적 결정). 하이퍼파라미터 개선폭의 부트스트랩 CI가 0에 가깝게 걸쳐
  있다는 점을 볼 때, 실제 배포 전 반드시 별도 validation split 또는 추가 데이터로 재검증해야 한다.
- 절대적 성능 자체가 낮다(NDCG@10 ≈ 0.01 수준) — 데이터셋의 개인화 신호가 원래 약할 가능성을
  배제할 수 없다.
- purchase 이벤트만 정답으로 사용해 평가 대상이 희소하다(train 유저 19,930명 중 1,463명, 7.3%).
- `score_map`(`page_view=1, add_to_cart=3, checkout=4, purchase=5`) 값 자체의 재설계는 시도하지
  않았다 — BM25 실험과 별개로, 언젠가 validation split이 갖춰지면 검토할 수 있는 영역으로 남겨둔다.

---

## 변경된 파일 목록

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `src/baselines/ALS/als_model.py` | 수정 | `recommend_all()` deprecated 버그 수정, `regularization` 배선, `apply_weighting()` 추가(no-op 기본값) |
| `configs/ALS/params.yaml` | 수정 | `als:` 블록 튜닝 결과 반영 (factors/alpha/regularization) |
| `notebooks/20260703_ML_cold_threshold_search.ipynb` | 신규 | Cold threshold 탐색 (구 하이퍼파라미터 → 신규 하이퍼파라미터 → K=20 재확인 3회 재실행) |
| `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb` | 신규 | 하이퍼파라미터 80조합 그리드서치 |
| `notebooks/20260703_ML_bm25_weighting_redesign.ipynb` | 신규 | BM25/TF-IDF 가중치 재설계 검토 (No-Go) |
