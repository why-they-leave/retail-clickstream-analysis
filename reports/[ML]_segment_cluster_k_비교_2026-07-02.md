# Segment Cluster K 비교 리포트

## 분석 목적

Issue #16의 feature 기반 segment assignment에서 사용할 KMeans cluster 수를 검토한다.

이번 비교는 `k=3~9` 후보에 대해 아래 기준을 함께 확인했다.

- `inertia`: cluster 내부 응집도
- `silhouette`: cluster 간 분리도
- `min_segment_ratio`, `max_segment_ratio`: segment size balance

## 입력 데이터

- 입력 파일: `data/processed/segment_features_all_customers.csv`
- 평가 스크립트: `src/features/evaluate_segment_clusters.py`
- 설정 파일: `configs/segment/params.yaml`
- 사용 피처:
  - `page_view_count`
  - `atc_rate`
  - `order_count`
  - `purchase_per_session`
  - `total_spend_log`
  - `recency_session_days`
  - `recency_order_days`
  - `view_purchase_category_match`
  - `dominant_view_category_ratio`
  - `dominant_purchase_category_ratio`

## 실행 명령

```bash
uv run python src/features/build_segment_features.py
uv run python src/features/evaluate_segment_clusters.py
```

## 결과

| k | inertia | silhouette | min segment | max segment | min ratio | max ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 118548.3878 | 0.2161 | 3,732 | 8,326 | 18.66% | 41.63% |
| 4 | 106628.9875 | 0.2097 | 3,732 | 6,434 | 18.66% | 32.17% |
| 5 | 99073.3524 | 0.2066 | 2,034 | 5,892 | 10.17% | 29.46% |
| 6 | 93031.2846 | 0.2039 | 1,748 | 4,669 | 8.74% | 23.34% |
| 7 | 87165.4961 | 0.1981 | 978 | 4,621 | 4.89% | 23.10% |
| 8 | 83027.3271 | 0.1909 | 976 | 3,927 | 4.88% | 19.63% |
| 9 | 79367.6059 | 0.1923 | 967 | 3,602 | 4.84% | 18.01% |

## 해석

### 1. Silhouette score는 전반적으로 낮다

모든 후보 k에서 silhouette score가 `0.19~0.22` 수준이다. 이는 고객들이 명확히 분리되는 자연 군집을 갖고 있다고 보기 어렵다는 뜻이다.

따라서 이번 KMeans 결과는 “통계적으로 뚜렷하게 발견된 고객군”이라기보다는, 고객 행동 피처를 기반으로 고객을 해석 가능한 크기의 그룹으로 압축한 1차 baseline으로 해석해야 한다.

### 2. k=3~4는 너무 거칠다

`k=3`은 silhouette가 가장 높지만 최대 segment가 전체의 `41.63%`를 차지한다. `k=4`도 최대 segment가 `32.17%`로 크다. 두 경우 모두 고객군이 너무 크게 묶여 LLM naming이나 Streamlit 데모에서 세그먼트 차이를 설명하기 어렵다.

### 3. k=7 이상은 작은 segment가 생긴다

`k=7`부터 최소 segment 비율이 `5%` 미만으로 내려간다. 너무 작은 segment는 naming이 불안정하고, Full/US 비교나 데모 화면에서 다루기 애매할 수 있다.

### 4. k=6은 실용적인 균형점이다

`k=6`은 silhouette가 최고는 아니지만, 최소 segment `8.74%`, 최대 segment `23.34%`로 size balance가 비교적 안정적이다. segment 수가 너무 적지도 많지도 않아 LLM naming과 Streamlit 데모에서 다루기 좋다.

## 1차 결정

Issue #16의 1차 KMeans baseline은 `k=6`으로 유지한다.

단, 이 결정은 “최적의 자연 군집 수”라기보다 아래 기준을 만족하는 실용적 선택이다.

- segment size가 과도하게 쏠리지 않음
- 5% 미만의 너무 작은 segment가 없음
- LLM naming과 데모 화면에서 설명 가능한 수준의 segment 수
- Full/US 비교 시 같은 segment definition을 적용하기 쉬움

## 한계 및 후속 검토

- silhouette score가 낮기 때문에 KMeans cluster boundary를 강하게 해석하면 안 된다.
- 후속 실험에서는 아래 방법과 비교가 필요하다.
  - GMM
  - hierarchical clustering
  - rule-based segmentation
  - 구매자/비구매자를 먼저 나눈 뒤 각 그룹 안에서 clustering
- Streamlit 데모에서는 `segment_id` 자체보다 `segment_summary`의 행동적 특징을 중심으로 설명하는 것이 안전하다.

## 관련 산출물

- `data/processed/segment_cluster_evaluation.csv`
- `data/processed/segment_cluster_sizes.csv`
- `data/processed/customer_segments_all_customers.csv`
- `data/processed/customer_segments_us_customers.csv`
- `data/processed/segment_summary_all_customers.csv`
- `data/processed/segment_summary_us_customers.csv`
