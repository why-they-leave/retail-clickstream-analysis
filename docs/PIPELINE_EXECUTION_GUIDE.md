# Pipeline Execution Guide — 페르소나/세그먼트 실행 커맨드

**대상**: `src/persona/`, `src/features/`의 페르소나·세그먼트 생성 파이프라인
**설계 배경**: `docs/WHY_SEGMENTS.md`(왜 이 작업을 하는지), `docs/SEGMENT_ASSIGNMENT_DESIGN.md`(v2 설계 상세)
**이 문서의 역할**: 위 두 문서는 "왜/무엇을"만 다루고 "정확히 어떤 커맨드로 실행하는지"는 다루지 않는다. 이 문서는 실행 커맨드만 다룬다.

---

## 사전 조건

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml`의 `[tool.hatch.build.targets.wheel] packages = ["src"]` 설정 때문에 editable install 이후에는 **`src.xxx` 형태의 임포트는 어느 cwd에서 실행하든 동작한다.** 단, 아래에서 `cwd 필수` 로 표시한 스크립트는 패키지 접두사 없는(bare) 임포트를 쓰기 때문에 반드시 지정된 디렉터리에서 실행해야 한다 (다른 cwd에서 실행하면 `ModuleNotFoundError`).

LLM 호출이 있는 단계는 `.env`에 `UPSTAGE_API_KEY`가 필요하다 (`.env.example` 참고).

---

## v1: LLM 직접 생성 페르소나 (GPLR 논문 방식, Issue #18 v1 vs v2 비교용)

**cwd 필수: `src/`** — `import datasets.MBA`, `from llm_connector...`, `from persona.config...` 가 모두 bare 임포트라 `src/`를 사이트 루트로 잡고 실행해야 한다. `-m` 플래그로 실행할 것.

```bash
cd src

# 1. 페르소나 후보 생성 (Step1 40회 → Step2 8회 → Step3 1회, 내부 자동 순차 실행)
python3 -m persona.generation --config ../configs/persona/params.yaml
# --config 생략 시 configs/persona/params.yaml이 기본값으로 적용된다
# US-only만 필요하면: --config ../configs/persona/params_us.yaml

# 2. 생성된 페르소나를 유저(+상품)에 배정 — 1의 산출물(final_personas.json)이 있어야 실행 가능
python3 -m persona.label_users_only --config ../configs/persona/params.yaml   # 유저만 (#18 비교용, LLM 호출 절약)
# 또는
python3 -m persona.labeling                                                    # 유저 + 상품 모두
```

> **주의**: 각 스크립트 파일 상단 docstring에는 `실행: python3 generation.py`라고 적혀 있지만, 실제로는 `ModuleNotFoundError: No module named 'datasets'`로 실패한다(확인됨). 위처럼 `-m` 플래그로 실행해야 한다. docstring이 최신화되지 않은 상태다.

**산출물**: `data/interim/funnel_persona_gen/final_personas.json`, `user_persona_labels_v1.json` 등 (US 트랙은 `funnel_persona_gen_us/`)

---

## v2: 통계적 세그먼트 + LLM 네이밍 (현재 production, LightGCN이 실제로 쓰는 파이프라인)

### 전체 기간 (production)

```bash
# 1. raw → 고객 단위 집계 피처 (cwd 필수: src/features — bare 임포트)
cd src/features
python3 run_customer_pipeline.py
#   내부: preprocess_joins.py → build_customer_features.py
#   출력: data/processed/customer_features_all_customers.csv

# 2. 집계 피처 → 클러스터링 입력 피처 (cwd 무관, 관례상 src/features에서)
python3 build_segment_features.py
#   출력: data/processed/segment_features_all_customers.csv

# 3. KMeans로 segment_id 부여 — 통계적 클러스터링, LLM 미사용 (cwd 무관)
python3 assign_segments.py
#   출력: customer_segments_all_customers.csv, segment_summary_all_customers.csv

cd ../persona

# 4. LLM으로 segment_id별 이름/설명만 부여 (개별 유저 배정 아님, cwd 무관)
python3 segment_naming.py --n-runs 5
#   출력: experiments/segment_naming_v2/run_<날짜>_1..5/segment_personas.json
#   (canonical 파일은 건드리지 않음 — 후보만 생성)

# 5. 사람이 후보를 검토해 experiments/segment_naming_v2/CHOICES.md에 채택 사유 기록 후 승격
python3 segment_naming.py --promote run_2026-07-06_3
#   API 호출 없음(파일 복사만). 출력: data/processed/segment_personas_v2.json

# 6. segment_id 기준으로 고객 ↔ 이름 병합 (API 호출 없음)
python3 segment_naming.py --merge-customers
#   출력: data/processed/customer_segments_labeled_all_customers.csv
```

### train 기간 한정 (Issue #31, LightGCN 전용 — 위 1~4를 CUTOFF_DATE 이전 데이터로 자동 실행)

```bash
# cwd 무관 (src.-prefixed 임포트만 사용), 관례상 src/features에서 실행
cd src/features
python3 build_train_only_segments.py
#   CUTOFF_DATE = 2025-08-01 (ALS configs/ALS/params.yaml의 split_date와 동일해야 함)
#   최종 출력: data/processed/customer_segments_labeled_train_only.csv
#   ↳ LightGCN tri-graph의 u2p(유저-세그먼트) 입력으로 사용됨
```

---

## 요약 표

| 트랙 | 실행 순서 | cwd | 최종 산출물 | 용도 |
|------|-----------|-----|-------------|------|
| v1 (GPLR 직접 생성) | `persona.generation` → `persona.label_users_only`/`persona.labeling` | `src/` (`-m` 필수) | `user_persona_labels_v1.json` | #18 v1 vs v2 비교 |
| v2 (통계 클러스터+LLM 네이밍, 전체 기간) | `run_customer_pipeline.py` → `build_segment_features.py` → `assign_segments.py` → `segment_naming.py`(4단계) | 1번만 `src/features` 필수, 나머지 무관 | `customer_segments_labeled_all_customers.csv` | production |
| v2 train-only (#31) | `build_train_only_segments.py` (원클릭) | 무관 | `customer_segments_labeled_train_only.csv` | **LightGCN tri-graph 입력** |

## 관련 문서

- 왜 이 작업을 하는지, v1이 왜 실패했는지: `docs/WHY_SEGMENTS.md`
- v2 설계 상세(config 관리 범위, 알려진 제약사항): `docs/SEGMENT_ASSIGNMENT_DESIGN.md`
- train-only 버전 도입 배경 및 검증 결과: `reports/[ML]_LightGCN용_train_전용_세그먼트_재계산_2026-07-05.md`
- LightGCN tri-graph가 이 산출물을 어떻게 쓰는지: `reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`
