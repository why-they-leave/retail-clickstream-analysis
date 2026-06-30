# [ML] #8 llm_connector 리팩토링 및 프롬프트 개선 — 2026-06-30

## 작업 요약

`Collector.py`에 혼재하던 책임을 분리하고, `describe_user()`에 집계 피처 기반 행동 컨텍스트를 추가했다.

---

## 변경된 파일 목록

| 파일 | 변경 유형 | 내용 |
|------|-----------|------|
| `src/llm_connector/formatter.py` | 신규 | 텍스트 변환 전담 (`describe_user`, `describe_users`) |
| `src/llm_connector/client.py` | 신규 | LLM 호출 + 병렬 처리 중앙화 (`call_llm`, `run_parallel`) |
| `src/llm_connector/parser.py` | 신규 | 응답 파싱 전담 (`parse_user_response`, `parse_item_response`) |
| `src/llm_connector/Collector.py` | 수정 | 하위 호환성 shim으로 축소 (기존 import 유지) |
| `src/persona/labeling.py` | 수정 | 새 모듈 import + `customer_features` 연결 + 프롬프트 강화 |
| `src/persona/generation.py` | 수정 | 새 모듈 import + `run_parallel` 적용 |
| `docs/PROMPT_CATALOG.md` | 신규 | 전체 프롬프트 원문 + 번역 문서 |
| `configs/persona/params.yaml` | 신규 | 전체 데이터 실행 파라미터 |
| `configs/persona/params_us.yaml` | 신규 | US-only 실행 파라미터 (sample ratio 10%) |
| `configs/persona/run_local.yaml` | 신규 | 전체 데이터 로컬 실행 설정 (gitignore) |
| `configs/persona/run_local_us.yaml` | 신규 | US-only 로컬 실행 설정 (gitignore) |
| `.coderabbit.yaml` | 신규 | ecommerce 레포에서 복사 |

---

## 구조 변경 상세

### Before

```text
src/llm_connector/
└── Collector.py       ← 텍스트 변환 + LLM 호출 + 파싱 혼재

src/persona/
├── generation.py      ← ThreadPoolExecutor 직접 관리
└── labeling.py        ← ThreadPoolExecutor 직접 관리 + call_llm 중복
```

### After

```text
src/llm_connector/
├── formatter.py       ← describe_user() (텍스트 변환 전담)
├── client.py          ← call_llm(), run_parallel() (호출 + 병렬 중앙화)
├── parser.py          ← parse_user_response(), parse_item_response()
└── Collector.py       ← shim (기존 import 호환용)

src/persona/
├── generation.py      ← run_parallel() 위임
└── labeling.py        ← run_parallel() 위임 + customer_features 연결
```

---

## 주요 변경사항

### 1. `describe_user()` 확장

`customer_features=None`이면 기존 동작 유지 (하위 호환).
`customer_features`가 전달되면 아래 블록 추가:

```text
Behavioral context:
- Sessions: 15
- Page views: 45, Add-to-carts: 8 (ATC/PV: 17.8%)
- Orders: 3, Total spend: $234.50, Average order value: $78.17
- Last session: 32 days ago, Last order: 45 days ago
- Most viewed category: Books, Most purchased category: Beauty
```

결측값은 `N/A`로 표시 (`recency_order_days`, `avg_order_value`, `top_purchase_category`, ATC/PV).

### 2. 프롬프트 강화 (labeling.py)

아래 두 구문을 유저/상품 라벨링 프롬프트에 추가:

```text
Select only exact persona names from the given list.
Do not rename, paraphrase, abbreviate, or create new personas.

Only output valid JSON. No markdown. No explanation.
```

### 3. `run_parallel()` 병렬 처리 중앙화

`labeling.py`, `generation.py`에 중복 구현된 `ThreadPoolExecutor` 패턴을 `client.run_parallel(fn, tasks_args, max_workers, desc)`로 통일.

---

## 검토 순서

### 1단계 — 모듈 구조 확인

```text
src/llm_connector/formatter.py   ← describe_user() 시그니처 및 Behavioral context 블록
src/llm_connector/client.py      ← call_llm(), run_parallel() 구현
src/llm_connector/parser.py      ← parse_user_response(), parse_item_response()
src/llm_connector/Collector.py   ← shim 확인 (import 목록만 있어야 함)
```

### 2단계 — labeling.py 핵심 변경점

- `import` 상단: `from llm_connector.client import call_llm, run_parallel` 확인
- `assign_user_label()`: `customer_features=None` 파라미터 추가 확인
- `make_user_prompts()` / `make_item_prompts()`: JSON + exact name 구문 추가 확인
- `main()`: `customer_features` CSV 로딩 코드 (파일 없으면 경고 후 기본 모드)
- `main()`: `run_parallel(assign_user_label, tasks_args, ...)` 호출 확인

### 3단계 — generation.py 핵심 변경점

- `import` 상단: 새 모듈 import 확인
- `step1()`, `step2()`: `run_parallel()` 위임 확인
- `_step1_single()`, `_step2_single()`: 단일 실행 함수로 분리 확인

### 4단계 — 프롬프트 문서

```text
docs/PROMPT_CATALOG.md   ← 원문/번역 대조 확인
```

### 5단계 — 샘플 출력 검증 (선택)

```python
import pandas as pd
from src.llm_connector.formatter import describe_user

df = pd.read_csv("data/interim/funnel_mba_format.csv")
grouped = df.groupby(["CustomerID", "Itemname"]).agg({"Quantity": "sum"}).reset_index()
cf = pd.read_csv("data/processed/customer_features_all_customers.csv")

uid = grouped["CustomerID"].iloc[0]
print(describe_user(uid, grouped, customer_features=cf))   # Behavioral context 포함
print(describe_user(uid, grouped))                          # 기존 형식
```

---

## 주의사항

- `labeling.py`의 `item_names`는 여전히 `global` 변수 — `assign_user_label` 내부에서 참조함
- `run_local.yaml`, `run_local_us.yaml`은 gitignore 처리됨 (커밋 대상 아님)
- `customer_features_all_customers.csv`가 없으면 경고 출력 후 M1 방식으로 실행됨
