# [ML] LightGCN용 tri-graph 데이터 파이프라인 구축 — 2026-07-06

## 분석 목적

Issue #28(LightGCN 베이스라인 구현)의 데이터 준비 단계(Issue #29)로, `src/baselines/lgcn3/`의 LightGCN_tri 모델이 요구하는 tri-graph 입력(유저-상품-세그먼트 그래프)을 이 프로젝트의 실제 데이터로 만드는 작업이다. ALS(#7)와 공정하게 비교하기 위해 같은 train/test 경계, 같은 평가 기준을 지키는 것을 핵심 제약으로 삼았다.

## 데이터셋 버전 및 기간

- 고객 카탈로그: `data/raw/customers.csv` (20,000명)
- 상품 카탈로그: `data/raw/products.csv` (1,197개)
- 상호작용: `data/interim/sessions_events_products.csv`, `data/interim/orders_items_products.csv`
- 세그먼트: `data/processed/customer_segments_labeled_train_only.csv` (#31 산출물, train 기간 한정 재계산본)
- 분할 기준: `timestamp < 2025-08-01` = train, 그 이후 = valid (ALS `configs/ALS/params.yaml`의 `split_date`와 동일)

## 방법론

```text
customers.csv / products.csv
→ build_id_encoding()                         : user_id/item_id → 0-base 정수 인덱스 (ALS와 공용)
sessions_events_products.csv / orders_items_products.csv
→ CUTOFF_DATE(2025-08-01) 기준 train/valid 분할
→ build_u2t_mapping()  (train)                : 유저-상품 상호작용, event_type 파라미터화
→ build_u2t_mapping()  (valid, 구매만 고정)     : ALS 평가 기준(test_pairs)과 동일하게 구매만 정답
→ build_u2p_mapping()                          : customer_segments_labeled_train_only.csv → 단일 라벨
→ build_t2p_mapping()                          : train 구매 데이터로 Lift 계산 → 다중 연결(eric 정정)
→ tri_graph_{uidx2tidx_train,uidx2tidx_valid,uidx2pidx,tidx2pidx}.json
```

관련 코드: `src/datasets/make_lgcn_graph.py`(신규), `src/utils/id_encoding.py`(신규, ALS와 공용), `src/baselines/lgcn3/save_recommendations.py`(신규)

**핵심 설계 결정 3가지:**

1. **u2t(학습용 유저-상품 엣지)는 하나로 확정하지 않고 실험 파라미터로 남김.** ALS는 `page_view=1/add_to_cart=3/purchase=5`처럼 가중치를 주지만, LightGCN 원 코드는 가중치 없는 이분 그래프만 지원한다. 어떤 `event_type` 조합이 최선인지는 #30에서 Hit Ratio@20 기준으로 실험적으로 정하기로 했다 (팀 논의 결과).
2. **valid(평가용) u2t는 실험 대상이 아니라 고정.** ALS의 `test_pairs`가 "test 기간 구매만" 정답으로 삼는 것과 동일하게, LightGCN의 평가 정답도 구매만으로 고정해야 두 모델이 같은 시험을 보게 된다.
3. **item→segment(t2p)는 Lift 기반 다중 연결.** 초기에는 "가장 높은 세그먼트 1개만 연결"로 정리했었으나, 팀원(eric)이 "lift 기준 자체에 단일 연결 제약이 없다"고 정정했고, 코드 검토 결과(`labeling.py`의 v1 프롬프트, `tri_graph_tidx2pidx.json` 리스트 포맷, `propagation_matrix_tri()`의 페어 리스트 입력)도 다중 연결을 전제로 설계돼 있었음을 확인했다. 최종적으로 `lift > 1.15`인 세그먼트 전부와 연결하고, lift 값 자체를 그래프 엣지 가중치로 반영했다.

**모델 코드 수정 (그래프 구성 유틸에 한정, 학습 로직은 미변경):**

- `src/baselines/lgcn3/dense2sparse.py`: `propagation_matrix_tri()`가 `(item, persona)` 2-튜플(기존)과 `(item, persona, weight)` 3-튜플(신규, lift 가중치)을 모두 받도록 확장. weight 생략 시 1.0으로 처리해 하위 호환.
- `src/baselines/lgcn3/read_data.py`: t2p 파싱 시 `persona_id`(기존) 또는 `[persona_id, weight]`(신규) 둘 다 지원. `DIR` 경로도 `data/` 최상위 대신 `data/processed/`를 가리키도록 수정(프로젝트 데이터 폴더 관례 준수, #30과 공유하는 코드라 이슈에 별도 기록).

## 평가 지표

지도학습 평가가 아니므로, 산출물 무결성과 그래프 통계로 확인했다.

| 지표 | 값 |
|------|-----|
| user_num / item_num | 20,000 / 1,197 |
| train u2t — 상호작용 있는 유저 수 | 19,930 / 20,000 |
| valid u2t(구매만) — 정답 있는 유저 수 | 1,465 / 20,000 |
| t2p — lift 조건 충족해 연결된 상품 수 | 1,143 / 1,197 |
| 단위 테스트 | 9/9 통과 (`tests/test_lgcn_graph.py`) |
| 기존 회귀 테스트 | 28/28 통과 |

## 분석 결과

**핵심 발견:**
- u2t/u2p/t2p 세 그래프 모두 `read_data.py`가 요구하는 형식(모든 유저/상품이 키로 존재, 값은 빈 리스트 허용)을 만족하도록 생성했다 — `user_num`/`item_num`이 딕셔너리 길이로 결정되는 구조라 이 부분을 놓치면 조용히 잘못된 값이 나올 위험이 있었다.
- t2p 다중 연결 검증 결과, 1,197개 상품 중 1,143개(95.5%)가 최소 1개 이상의 세그먼트와 연결됐고, 상당수는 2개 이상 세그먼트에 동시 연결됐다 — 단일 연결이었다면 이 정보가 손실됐을 부분이다.
- `dense2sparse.py`/`read_data.py` 수정은 tensorflow가 설치되지 않아 실제 실행 검증을 못 했다 — 문법 확인과 순수 파이썬 로직(가중치 언패킹)만 검토했고, 실제 그래프 텐서 생성은 #30에서 확인해야 한다.

**해석:** 이번 작업으로 LightGCN 학습에 필요한 입력은 다 준비됐지만, u2t의 최적 `event_type` 조합과 t2p의 정확한 threshold/최소 구매 건수 값은 아직 실험적으로 확정되지 않은 상태다 — 이는 의도된 것으로, #30에서 실제 학습·평가를 반복하며 정할 사안이다.

## 방법론적 제약 및 한계

- t2p의 `lift > 1.15`, 최소 구매 건수 `5건` 기준은 이론적 근거보다는 합리적인 시작값이다. 상품 카탈로그가 1,197개로 크지 않아 threshold를 더 튜닝할 여지가 있다.
- u2t event_type 조합(page_view+add_to_cart+구매 vs 일부만)에 따라 그래프 밀도가 크게 달라질 수 있고, 이게 LightGCN 성능에 어떤 영향을 주는지는 아직 실험하지 않았다.
- 모델 내부 코드(`dense2sparse.py`, `read_data.py`) 수정분은 tensorflow 미설치로 end-to-end 실행 검증이 안 된 상태다.

## 관련 산출물

- `src/datasets/make_lgcn_graph.py` (신규 — tri-graph 생성 오케스트레이션)
- `src/utils/id_encoding.py` (신규 — ALS/LightGCN 공용 ID 인코딩)
- `src/baselines/ALS/als_model.py` (`build_sparse_matrix()`가 공용 인코딩 함수를 쓰도록 리팩터링)
- `src/baselines/lgcn3/dense2sparse.py`, `read_data.py` (t2p 가중치 지원, DIR 경로 수정)
- `src/baselines/lgcn3/save_recommendations.py` (신규 — `PRED_MAIN_RECOMMEND.csv` 저장, `model_type="LightGCN"`)
- `tests/test_lgcn_graph.py` (신규 — 단위 테스트 9건)
- `data/processed/tri_graph_uidx2tidx_train.json`, `tri_graph_uidx2tidx_valid.json`, `tri_graph_uidx2pidx.json`, `tri_graph_tidx2pidx.json` (gitignore 대상 로컬 산출물)

## 권장 다음 단계

- [ ] #30(eric): `tensorflow` 설치 후 `dense2sparse.py`/`read_data.py` 실제 실행 검증
- [ ] #30: u2t event_type 조합별 실험 (전부 포함 vs add_to_cart+구매 vs 구매만) → Hit Ratio@20 비교
- [ ] #30: `run_lightgcn.py` 작성 및 학습·평가, ALS와 성능 비교 리포트 작성
