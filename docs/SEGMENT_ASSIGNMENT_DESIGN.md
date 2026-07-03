# Segment Assignment Design

## 목적

Issue #16에서는 Issue #15에서 확정한 고객 단위 피처를 사용해 clustering 기반 `segment_id`를 생성하고, LLM naming/summarization에 넘길 `segment_summary`를 만든다.

전체 흐름은 아래와 같다.

```text
customer_features_all_customers.csv
→ segment_features_all_customers.csv
→ customer_segments_all_customers.csv
→ segment_summary_all_customers.csv
```

> **Issue #23 반영**: #16 초기 구현은 Full(all_customers)과 US-only(us_customers)를 병행 생성했지만, #4에서 이미 "Full과 US-only 간 유의미한 차이 없음"이 확인되어 #23에서 US-only 트랙을 제거했다. 이후 파이프라인은 Full 데이터 단일 트랙으로만 동작한다.

## Config 관리 범위

1차 구현에서는 `configs/segment/params.yaml`에 **실험 결과를 바꾸는 값**만 둔다.

예:

```yaml
segment_assignment:
  method: kmeans
  n_clusters: 6
  random_state: 42
  n_init: 20
  input_features:
    - page_view_count
    - atc_rate
    - order_count
    - purchase_per_session
    - total_spend_log
    - recency_session_days
    - recency_order_days
    - view_purchase_category_match
    - dominant_view_category_ratio
    - dominant_purchase_category_ratio
```

위 값들은 cluster 결과와 해석을 직접 바꾸므로 config로 관리한다.

## 데이터 경로 관리 원칙

1차 구현에서는 데이터 입출력 경로를 config에 넣지 않고, 코드의 표준 파이프라인 경로로 유지한다.

예:

```text
data/processed/customer_features_all_customers.csv
data/processed/segment_features_all_customers.csv
data/processed/customer_segments_all_customers.csv
data/processed/segment_summary_all_customers.csv
```

이 경로들은 실험 조건이라기보다 프로젝트 내부의 고정된 파이프라인 산출물 위치에 가깝다. 현재 단계에서 경로까지 config로 분리하면, 자주 바꾸는 실험 조건과 고정 경로가 섞여 config가 불필요하게 커질 수 있다.

따라서 현재 기준은 아래처럼 둔다.

| 구분 | 관리 위치 | 예시 |
|------|-----------|------|
| 실험 조건 | config | `n_clusters`, `random_state`, `n_init`, `input_features` |
| 표준 파일 경로 | code | `data/processed/segment_features_all_customers.csv` |

## 향후 DB/Streamlit 연동 시 확장

Streamlit 데모에서는 raw 데이터를 SQLite에 저장하는 구조를 고려한다. 이 단계에서는 CSV 기반 pipeline을 먼저 완성하고, Streamlit/SQLite 연동 이슈에서 저장소만 DB로 확장한다.

예상 흐름:

```text
Streamlit raw data input
→ SQLite raw/interim/customer_features table
→ segment_features table
→ customer_segments table
→ segment_summary table
→ Streamlit SELECT
```

이때는 아래 항목을 별도 config 또는 app settings로 분리하는 것을 검토한다.

- SQLite DB path
- table names
- model artifact path
- model version
- fit/predict mode

즉 #16에서는 CSV 산출물을 안정적으로 만들고, DB 저장/조회는 후속 Streamlit 구현 단계에서 붙인다. 다만 `build_segment_features.py`, `assign_segments.py`의 핵심 로직은 함수로 분리해 Streamlit에서도 재사용할 수 있게 유지한다.
