# [ML] Segment Naming v2 (LLM 기반 세그먼트 라벨링) — 2026-07-03

## 분석 목적

Issue #17: Issue #16에서 KMeans로 확정한 6개 behavior segment(`segment_id` 0~5)에 대해, `segment_summary` 집계 통계만 근거로 LLM이 사람이 읽을 수 있는 이름과 설명을 생성하는 naming/summarization 파이프라인을 구현한다.

v1(`persona/generation.py`, `persona/labeling.py`)과 달리 LLM은 persona taxonomy를 창작하거나 개별 고객을 분류하지 않는다. `segment_id`는 이미 #16에서 KMeans가 확정했고, LLM은 그 결과를 사람이 읽을 말로 번역하는 라벨러 역할로 제한된다.

## 데이터셋 버전 및 기간

- 원본: `data/raw/{customers,sessions,events,orders,order_items,products}.csv` — 합성 리테일 클릭스트림 데이터셋(날짜 기반 기간 개념 없음)
- 입력: `data/processed/segment_summary_all_customers.csv` (Issue #16 산출물, 전체 고객 20,000명 기준 집계)
- Full(20,000명)과 US(3,648명)는 all_customers 기준으로 fit한 동일 KMeans pipeline을 공유하므로, `segment_id`가 같은 클러스터를 가리킨다(`docs/SEGMENT_ASSIGNMENT_DESIGN.md`).

## 분할 전략

해당 사항 없음(train/val/test 분할이 아님). LLM 호출 단위는 "세그먼트 1개당 1회"이고, 개별 고객 row는 입력에 포함하지 않는다. `segment_id` 배정 자체는 #16에서 이미 끝난 상태라 이번 단계에서 새로 발생하는 데이터 누수 지점은 없다.

naming은 all_customers summary 기준으로 1회만 수행하고, 결과를 `segment_id` 기준으로 US customer_segments 테이블에도 그대로 병합한다(US summary로 별도 LLM 호출을 하지 않음 — Full/US가 동일 segment 정의를 공유하기 때문).

## 방법론

```text
segment_summary_all_customers.csv
→ format_segment_summary()      : row → 자연어 텍스트 (결측치 → "none")
→ build_user_prompt()           : system/user 프롬프트 조립
→ call_llm() (Upstage solar-pro)
→ validate_naming_response()    : schema 검증, 실패 시 최대 2회 재시도 → NAMING_FAILED
→ check_demographic_language()  : demographic/lifestyle 키워드 소프트 체크(경고만)
→ segment_personas_v2.json 저장
→ merge_personas_with_customers(): segment_id 기준 customer_segments 병합
```

관련 코드: `src/persona/segment_naming.py`
프롬프트 원문/번역: `docs/PROMPT_CATALOG.md` §3

## 평가 지표

표준 CV/정확도 지표가 적용되는 지도학습 task가 아니므로, 아래 기준으로 파이프라인 신뢰도를 확인했다.

| 지표 | 값 |
|------|-----|
| schema 검증 통과율 | 6/6 (100.0%) |
| 재시도 발생 횟수 | 0회 (전부 1차 시도 성공) |
| demographic/lifestyle 키워드 경고 | 0건 |
| customer_segments 병합 결측 | all 0/20,000행, us 0/3,648행 |

## 분석 결과

**핵심 발견:**
- 6개 세그먼트 모두 데모그래픽/라이프스타일 표현 없이 행동 기반 이름으로 생성됨 (예: `Browsing-Only with High Add-to-Cart`, `Inactive Frequent Purchasers`)
- 구매전환율 100%인 segment 4·5도 `view_purchase_category_match_rate`가 0.0%로 나와, "구매는 하지만 조회 카테고리와 구매 카테고리가 전혀 겹치지 않는" 패턴이 LLM 설명에도 정확히 반영됨
- segment_id 기준 병합이 Full/US 두 데이터셋 모두에서 결측 없이 성공 — 후속 Streamlit 연동에 바로 사용 가능한 상태

**수치 요약:**

| segment_id | customer_count | customer_ratio | purchaser_ratio | segment_name |
|---:|---:|---:|---:|---|
| 0 | 2,572 | 12.9% | 100.0% | Frequent-Viewing Purchasers with Diverse Interests |
| 1 | 3,726 | 18.6% | 0.0% | Browsing-Only with High Add-to-Cart |
| 2 | 1,758 | 8.8% | 99.7% | High-Engagement Low-Frequency Shoppers |
| 3 | 2,805 | 14.0% | 100.0% | Frequent-Viewing Purchasers with Narrow Category Focus |
| 4 | 4,639 | 23.2% | 100.0% | Inactive Frequent Purchasers |
| 5 | 4,500 | 22.5% | 100.0% | Low-Frequency Purchasers with Narrow Purchase Focus |

**해석:** 구매전환율(0% vs ~100%)이 segment를 가장 크게 가르는 축으로 보이며, 구매전환이 100%인 segment 내에서는 order_count·recency·category 집중도로 세분화되는 양상이다. 다만 아래 "한계점"에서 언급하듯 silhouette score가 낮아 이 구분을 뚜렷한 자연 군집으로 과잉 해석하면 안 된다.

## 핵심 가정 및 방법론적 제약

- Issue #16 기준 KMeans silhouette score가 `0.19~0.22` 수준으로 낮음(`reports/[ML]_segment_cluster_k_비교_2026-07-02.md`) — 프롬프트에 이 경고를 명시해 LLM이 과신하는 라벨을 피하도록 유도했다.
- LLM은 `segment_summary`의 집계 숫자만 근거로 사용하며, 개별 고객 raw 데이터·상품명은 입력에 없다.
- Full과 US customer_segments는 같은 `segment_id` 정의를 공유하므로 naming도 all_customers 기준 1회만 수행했다. US 고객군의 세부 특성이 all과 미묘하게 다를 수 있으나, 일관성·재현성을 우선해 US 전용 재검증은 하지 않는 설계 선택을 했다.

## 한계점 및 잠재 편향

- demographic/lifestyle 금지 규칙은 프롬프트 레벨 지시 + 소프트 키워드 경고(`check_demographic_language`)까지만 있고, 완전 자동 차단은 아니다. 새로운 라벨이 생성될 때마다 사람 검토를 권장한다.
- 실제 LLM 호출은 6건뿐이라 재시도/`NAMING_FAILED` 경로가 이번 실행에서는 발동하지 않았다(mock 클라이언트로만 별도 검증). 향후 재실행 시 API 응답 포맷이 흔들리는 케이스는 계속 모니터링이 필요하다.
- segment 4·5의 `view_purchase_category_match_rate = 0.0%`(구매전환 100%인데도 조회-구매 카테고리 완전 불일치)는 합성 데이터 특성일 가능성이 있다. 실제 서비스 데이터에 적용할 경우 이 패턴이 재현되는지 별도 확인이 필요하다.

## 관련 산출물

- `src/persona/segment_naming.py`
- `docs/PROMPT_CATALOG.md` (§3 Segment Naming v2)
- `data/processed/segment_personas_v2.json`
- `data/processed/segment_summary_{all,us}_customers.csv`
- `data/processed/customer_segments_{all,us}_customers.csv`

## 권장 다음 단계

- 리포트 검수 후 커밋/PR 여부 결정 (현재 미커밋 상태)
- Streamlit 데모에 `segment_name`/`description` 노출 연동 검토 (`docs/SEGMENT_ASSIGNMENT_DESIGN.md`의 향후 DB 확장 계획과 연결)
- `evidence` 항목이 실제 summary 값과 정확히 대응하는지 사람이 diff 체크하는 검토 프로세스 자동화 검토
