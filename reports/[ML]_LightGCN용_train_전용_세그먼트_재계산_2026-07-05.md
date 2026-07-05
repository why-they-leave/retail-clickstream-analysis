# [ML] LightGCN용 train 전용 세그먼트 재계산 — 2026-07-05

## 분석 목적

Issue #28(LightGCN 베이스라인 구현)에서 ALS(#7)와 성능을 비교하려면, 두 모델이 같은 train/test 경계로 평가돼야 한다. ALS는 `configs/ALS/params.yaml`의 `split_date: "2025-08-01"` 기준으로 train/test를 나누는데, 기존 세그먼트(`customer_segments_labeled_all_customers.csv`, #26)는 전체 기간(~2025-11-01) 데이터로 계산돼 있어 test 기간 정보가 이미 섞여 있었다.

이 이슈는 그 데이터 누수를 없애기 위해, 기존 세그먼트 파이프라인(#15~#18, #26)을 **train 기간(2025-08-01 이전) 데이터만으로 재실행**해 LightGCN 전용 세그먼트 산출물을 만드는 작업이다 (Issue #31).

## 데이터셋 버전 및 기간

- 원본: `data/interim/sessions_events_products.csv`, `data/interim/orders_items_products.csv` (전체 고객 20,000명)
- 필터링 기준: `timestamp < 2025-08-01` (ALS `split_date`와 동일)
- 필터링 결과: session_events 760,958 → 726,940행 (−4.5%), order_details 59,163 → 56,414행 (−4.6%)

## 방법론

```text
sessions_events_products.csv / orders_items_products.csv
→ CUTOFF_DATE(2025-08-01) 이전 데이터만 필터링
→ build_customer_features() (analysis_date=CUTOFF_DATE)   : 고객 단위 피처
→ add_segment_features()                                   : segment feature schema 적용
→ fit_segment_model() / assign_segment_ids()                : KMeans 재학습·재배정 (n_clusters=6)
→ build_segment_summary()                                   : 세그먼트별 집계 통계
→ label_segment() (Upstage LLM)                              : 세그먼트 naming (6회 호출)
→ merge_personas_with_customers() / validate_merged_segments()
→ customer_segments_labeled_train_only.csv
```

기존 파이프라인 함수(`src/features/build_customer_features.py`, `build_segment_features.py`, `assign_segments.py`, `src/persona/segment_naming.py`)를 재사용했고, 새 오케스트레이션 스크립트(`src/features/build_train_only_segments.py`)만 신규 작성했다. 프로덕션 산출물(전체 기간 기준)은 건드리지 않고 별도 파일(`*_train_only.*`)로 저장했다.

**부수적으로 발견/수정한 사전 존재 버그:** `assign_segments.py`가 `from segment_common import ...`처럼 `src.` 접두어 없이 import해서, 이 스크립트를 다른 모듈에서 import하면 `ModuleNotFoundError`가 났다 (#26 작업 때 `segment_naming.py`에서 찾은 것과 같은 종류의 버그). `src.features.segment_common`으로 수정했다.

## 평가 지표

표준 지도학습 평가가 아니므로, 파이프라인 무결성과 재계산 전후 비교로 신뢰도를 확인했다.

| 지표 | 값 |
|------|-----|
| customer_segments 검증 (row 수/segment_id 결측/cluster 수) | 통과 (rows=20,000, segments=6) |
| 세그먼트 병합 검증 (`validate_merged_segments`) | 통과 |
| LLM naming 호출 성공률 | 6/6 (100%) |
| segment_id 원시 일치율 (같은 customer_id 기준, train_only vs 프로덕션) | 13.6% |

## 분석 결과

**핵심 발견:**
- `segment_id` 원시 일치율이 13.6%로 낮게 나왔는데, 이는 재분류 자체가 잘못된 게 아니라 **KMeans가 실행마다 클러스터 번호를 임의로 새로 매기기 때문**이다. 세그먼트 이름으로 대조하면 `Non-Purchasing Browsers`(양쪽 다 존재), `Frequent Viewers with Consistent Purchases`(train_only 0번 = 프로덕션 3번) 등 상당수가 사실상 대응되는 그룹이다.
- train 기간(더 적은 데이터)만으로 재클러스터링하니 `Low-Engagement Non-Purchasers`(1,051명)라는, 프로덕션 6개 세그먼트 중 어디에도 명확히 대응되지 않는 새로운 그룹이 갈라져 나왔다.
- 이는 이전에 문서(`docs/SEGMENT_ASSIGNMENT_DESIGN.md`)에 이론적 우려로만 적혀 있던 "신규 데이터 유입 시 segment 불안정 가능성"이 실제로 재현된 사례다.

**수치 요약 — 세그먼트 분포 비교:**

| segment_id | train_only 인원 | 프로덕션 인원 | train_only 세그먼트명 | 프로덕션 세그먼트명 |
|---|---|---|---|---|
| 0 | 3,901 | 1,748 | Frequent Viewers with Consistent Purchases | High-Engagement Loyalists |
| 1 | 2,975 | 2,577 | Non-Purchasing Browsers | Non-Purchasing Browsers |
| 2 | 1,051 | 3,726 | Low-Engagement Non-Purchasers | High-Engagement Low-Frequency Purchasers |
| 3 | 6,019 | 4,669 | Frequent Browsers with Occasional Purchases | Frequent Viewers with Consistent Purchases |
| 4 | 1,979 | 4,498 | High-Engagement Occasional Purchasers | High-Engagement Non-Matching Purchasers |
| 5 | 4,075 | 2,782 | High-Engagement Repeat Purchasers | Low-Frequency Purchasers with Narrow Purchase Focus |

**해석:** segment_id 번호는 두 산출물 사이에서 서로 다른 의미를 가지므로 절대 번호로 비교하면 안 되고, 항상 세그먼트 이름으로 대조해야 한다는 게 이번 작업의 가장 중요한 확인 사항이다. LightGCN(#29, #30)은 이 리포트의 `customer_segments_labeled_train_only.csv`만 참조해야 하며, 프로덕션 세그먼트와 절대 혼용하면 안 된다.

## 방법론적 제약 및 한계

- KMeans 클러스터 라벨은 실행마다 임의로 재배정되므로, 이번 재계산 결과와 프로덕션 결과의 `segment_id` 숫자는 서로 호환되지 않는다. 세그먼트 정의를 비교/추적할 때는 반드시 `segment_name`을 기준으로 해야 한다.
- 세그먼트 번호가 실행마다 바뀌는 문제 자체(버전 관리·drift 감지)는 이번 작업 범위에서 해결하지 않았다 — 실제 운영 파이프라인의 재학습 주기가 정해지는 시점에 별도로 다룰 사안으로 남겨둔다(`docs/SEGMENT_ASSIGNMENT_DESIGN.md` 기존 기록과 동일한 결론).
- train 기간 데이터(전체 대비 세션 −4.5%, 주문 −4.6%)로도 세그먼트 분포가 꽤 다르게 나온 것은, 데이터 양보다 훈련 데이터에 포함된 "최근 고관여/구매 행동"의 비중이 줄어든 영향일 가능성이 높다 — 정확한 원인 분석은 하지 않았다.

## LightGCN 운영 시 유의사항 — 신규 로그로 세그먼트가 바뀌면?

LightGCN_tri는 **학습 시점의 그래프 스냅샷**(유저-상품-세그먼트 연결 구조)으로 임베딩을 한 번 학습하고 고정한다. 따라서:

- 학습 이후 신규 로그가 쌓여 어떤 유저의 `segment_id`가 바뀌더라도, **이미 학습된 모델은 이를 자동으로 반영하지 못한다.** 그 유저는 모델 안에서 여전히 학습 당시의 세그먼트에 연결된 채로 남는다.
- 반영하려면 그래프(세그먼트 매핑 포함)를 다시 만들고 모델을 재학습해야 한다 — 임베딩 하나만 부분 업데이트하는 방식은 이 구조상 불가능하다(그래프 전체를 다시 전파해야 새 임베딩이 나옴).
- 이는 이번 리포트에서 확인한 "세그먼트 번호가 재계산마다 바뀌는" 문제와 같은 근본 원인(세그먼트 정의가 재학습 시점마다 스냅샷으로 고정됨)에서 나온다.

**현재 스코프에서의 결론:** 이번 작업은 "ALS vs LightGCN, 1회 학습으로 성능 비교"하는 실험이지 상시 운영되는 서비스가 아니므로, 세그먼트 drift를 실시간 반영하는 문제는 지금 단계에서 다루지 않는다. `docs/SEGMENT_ASSIGNMENT_DESIGN.md`에 이미 명시된 대로, 실제 재학습 주기가 정해지는 시점에 모델 버전 관리·drift 감지 체계를 별도로 설계하는 것으로 남겨둔다.

## 관련 산출물

- `src/features/build_train_only_segments.py` (신규 오케스트레이션 스크립트)
- `src/features/build_customer_features.py` (`analysis_date` 파라미터화)
- `src/features/assign_segments.py` (import 버그 수정)
- `data/processed/customer_features_train_only.csv`
- `data/processed/segment_features_train_only.csv`
- `data/processed/customer_segments_train_only.csv`
- `data/processed/segment_summary_train_only.csv`
- `data/processed/segment_personas_train_only.json`
- `data/processed/customer_segments_labeled_train_only.csv` (LightGCN #29/#30에서 사용할 최종 산출물)

## 권장 다음 단계

- [ ] #29(데이터 파이프라인)에서 `customer_segments_labeled_train_only.csv`를 user→segment(u2p), item→segment(t2p) 매핑 생성에 사용
- [ ] 세그먼트 번호 불안정성(버전 관리) 문제는 실제 재학습 주기가 정해질 때 별도 이슈로 다룸
