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

> **업데이트(같은 날 후속 작업)**: 최초 실행 이후 LLM 응답이 호출마다 조금씩 달라지는 것을 확인해, `temperature=0` 고정과 실험 버전 관리 체계를 추가했다. 자세한 내용은 하단 "실험: naming 안정성 검증" 절 참고.

## 평가 지표

표준 CV/정확도 지표가 적용되는 지도학습 task가 아니므로, 아래 기준으로 파이프라인 신뢰도를 확인했다.

| 지표 | 값 |
|------|-----|
| schema 검증 통과율 | 6/6 (100.0%) |
| 재시도 발생 횟수 | 0회 (전부 1차 시도 성공) |
| demographic/lifestyle 키워드 경고 | 0건 (오탐 1건 확인 — 아래 "실험" 절 참고) |
| naming 안정성 (10회 반복, temperature=0) | 4/6 세그먼트 10/10 일치, 2/6 세그먼트 9/10 일치 |
| evidence-summary 기계 대조 불일치 | 0건 (채택된 `run_2026-07-03_1` 기준) |

## 분석 결과

**핵심 발견:**
- 6개 세그먼트 모두 데모그래픽/라이프스타일 표현 없이 행동 기반 이름으로 생성됨 (예: `Non-Purchasing Browsers`, `High-Engagement Non-Matching Purchasers`)
- 구매전환율 100%인 segment 4·5도 `view_purchase_category_match_rate`가 0.0%로 나와, "구매는 하지만 조회 카테고리와 구매 카테고리가 전혀 겹치지 않는" 패턴이 LLM 설명에도 정확히 반영됨 — 10회 반복 재현 검증과 evidence 기계 대조로 이 결과가 우연이 아님을 확인함
- `segment_id` 기준 `customer_segments` 테이블 병합(`merge_personas_with_customers()`)은 함수만 구현되어 있고 실제 파이프라인 어디에서도 호출되지 않음 — Full/US 산출물에 `segment_name`/`description` 컬럼이 아직 없음. 후속 작업 필요(아래 "권장 다음 단계" 참고)

> **업데이트 (2026-07-05, Issue #26)**: `merge_personas_with_customers()`가 `evidence`/`cautions`/`status`/`errors`까지 보존하도록 확장되고, `--merge-customers` CLI 옵션으로 실행 가능해졌다. 산출물: `data/processed/customer_segments_labeled_all_customers.csv`. 병합/검증 로직 단위 테스트 9건(`tests/test_persona_merge.py`) 추가, `segment_naming.py`의 `llm_connector` import 경로 버그(사전 존재)도 함께 수정.

**수치 요약:**

| segment_id | customer_count | customer_ratio | purchaser_ratio | segment_name |
|---:|---:|---:|---:|---|
| 0 | 2,572 | 12.9% | 100.0% | High-Engagement Loyalists |
| 1 | 3,726 | 18.6% | 0.0% | Non-Purchasing Browsers |
| 2 | 1,758 | 8.8% | 99.7% | High-Engagement Low-Frequency Purchasers |
| 3 | 2,805 | 14.0% | 100.0% | Frequent Viewers with Consistent Purchases |
| 4 | 4,639 | 23.2% | 100.0% | High-Engagement Non-Matching Purchasers |
| 5 | 4,500 | 22.5% | 100.0% | Low-Frequency Purchasers with Narrow Purchase Focus |

(위 `segment_name`은 `experiments/segment_naming_v2/run_2026-07-03_1` 채택본 기준. 최초 실행본과 이름이 다른 세그먼트가 있는 건 naming 자체의 LLM 비결정성 때문이며, `segment_id`가 가리키는 실제 고객 그룹(customer_count)은 동일하다.)

**해석:** 구매전환율(0% vs ~100%)이 segment를 가장 크게 가르는 축으로 보이며, 구매전환이 100%인 segment 내에서는 order_count·recency·category 집중도로 세분화되는 양상이다. 다만 아래 "한계점"에서 언급하듯 silhouette score가 낮아 이 구분을 뚜렷한 자연 군집으로 과잉 해석하면 안 된다.

## 실험: naming 안정성 검증 (temperature=0, 10 runs)

**문제**: 위 최초 실행 이후 파이프라인을 재호출해보니 `segment_name`이 매번 조금씩 다르게 나오는 것을 확인했다(예: segment 4가 `Inactive Frequent Purchasers` → `Frequent Non-Matching Browsers`로 변경). `call_llm()`이 `temperature`를 API 기본값에 맡기고 있어 재현성이 낮았다.

**조치**:
1. `src/llm_connector/client.py`의 `call_llm()`에 `temperature` 인자를 추가(기본값 `None` = 기존 동작 유지, v1 스크립트에는 영향 없음)
2. `src/persona/segment_naming.py`의 `label_segment()`에서 `temperature=0`을 기본값으로 사용
3. 실험 결과가 `data/processed/segment_personas_v2.json`(canonical 채택본)을 즉시 덮어쓰지 않도록, `experiments/segment_naming_v2/run_<날짜>_<순번>/segment_personas.json`에 매 실행을 별도 저장하는 구조를 추가하고, 검토 후 `--promote RUN_LABEL`로만 canonical 파일에 반영하도록 분리 (`experiments/segment_naming_v2/CHOICES.md`에 채택 기준·기록)

**결과**: `--dataset all --n-runs 10`으로 동일 입력에 10회 반복 호출한 결과, 6개 세그먼트 중 4개는 10/10 완전히 동일했고 나머지 2개도 9/10 동일(각 1개 run만 표현이 다름) — temperature=0 적용 전보다 안정성이 뚜렷하게 개선됨(완전한 결정성은 아님).

| segment_id | 10회 중 다수 의견 | 일치율 |
|---:|---|---:|
| 0 | High-Engagement Loyalists | 10/10 |
| 1 | Non-Purchasing Browsers | 10/10 |
| 2 | High-Engagement Low-Frequency Purchasers | 10/10 |
| 3 | Frequent Viewers with Consistent Purchases | 9/10 |
| 4 | High-Engagement Non-Matching Purchasers | 10/10 |
| 5 | Low-Frequency Purchasers with Narrow Purchase Focus | 9/10 |

`run_2026-07-03_1`이 6개 세그먼트 모두에서 다수 의견과 일치해 채택 심사 대상 1순위로 선정했다. 이 run의 `evidence` 항목 전체를 `segment_summary_all_customers.csv` 원본 수치와 `format_segment_summary()` 기준으로 기계적으로 대조한 결과, 근거 없는 숫자(할루시네이션) 0건으로 확인됨(일부 세그먼트에서 `dominant view category ratio` 1개 필드 누락은 있었으나 오류는 아님). demographic/lifestyle 키워드는 1건 검출(segment 3 "family status")됐으나, 실제 문장은 "family status에 대한 근거 없음"이라는 면책 문구라 위반이 아닌 것으로 판단했다. 이름 6개 전부 행동 기반 표현이고 broad/marketing 톤이 없어, `CHOICES.md` 채택 기준을 통과했다. **`run_2026-07-03_1`을 `--promote`로 canonical `data/processed/segment_personas_v2.json`에 최종 반영 완료.**

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
- `src/llm_connector/client.py` (`call_llm` temperature 지원 추가)
- `docs/PROMPT_CATALOG.md` (§3 Segment Naming v2)
- `data/processed/segment_personas_v2.json` (canonical 채택본 — `run_2026-07-03_1` 반영 완료)
- `data/processed/segment_summary_{all,us}_customers.csv`
- `data/processed/customer_segments_{all,us}_customers.csv`
- `data/processed/customer_segments_labeled_all_customers.csv` (Issue #26 — 고객별 세그먼트 라벨 테이블, gitignore 대상 로컬 산출물)
- `tests/test_persona_merge.py` (Issue #26 — 병합/검증 로직 단위 테스트 9건)
- `experiments/segment_naming_v2/run_2026-07-03_{1..10}/segment_personas.json` (안정성 검증용 반복 실행 결과)
- `experiments/segment_naming_v2/CHOICES.md` (채택 기준·기록)

## 권장 다음 단계

- [x] `experiments/segment_naming_v2/run_2026-07-03_1` 정성 검토 → `CHOICES.md`에 채택 기록 → `--promote run_2026-07-03_1`로 canonical 파일 갱신 (완료)
- [x] `merge_personas_with_customers()`를 실제로 호출하는 스크립트/CLI 옵션 추가 (완료, Issue #26) — `--merge-customers` 옵션으로 `evidence`/`cautions`/`status`/`errors`까지 보존한 `customer_segments_labeled_all_customers.csv` 생성
- [ ] PR #22 리뷰/머지 진행 (이슈 #17에 `Closes #17`로 연결됨)
- [ ] Streamlit 데모에 `segment_name`/`description` 노출 연동 검토 (`docs/SEGMENT_ASSIGNMENT_DESIGN.md`의 향후 DB 확장 계획과 연결)
- [ ] README 프로젝트 구조 설명에 `experiments/` 디렉터리 반영 (아직 미반영)
