# Prompt Catalog — GPLR 페르소나 파이프라인

> **관련 파일**: `src/persona/generation.py`, `src/persona/labeling.py`, `src/llm_connector/formatter.py`
> **모델**: Upstage Solar Pro (`solar-pro`)

---

## 1. 페르소나 생성 (Generation)

### Step 1 — 유저 구매이력 기반 후보 생성

**시스템 프롬프트**

```text
You are an assistant skilled at summarizing, capable of deducing
high-level consumer keywords based on a user's purchases.
```

**번역**

```text
당신은 요약에 능숙하고, 유저의 구매 내역을 바탕으로
고수준 소비자 키워드를 추론할 수 있는 어시스턴트입니다.
```

---

**유저 프롬프트**

```text
Take a deep breath and work according to the instructions step by step.
Now you will conduct a series of analyses on a Funnel E-commerce dataset.
This dataset contains data from an online retailer where each user's purchasing
transactions and the bought items are recorded. I will provide purchase information
from about 100 users; each user's data is grouped by ID and described in natural language.

Your task is to generate 20 representative and accurate user personas based on these
users' purchasing patterns. Optimize for two targets:

• High Coverage: The persona set should cover as many users as possible.
  Coverage = number of users that can be labeled with at least one persona.
• High Accuracy: Each persona must have a precise, non-ambiguous definition.

For each persona, provide a name and a concise one-sentence definition.
Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
2. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

User purchasing data:
{100명 유저 구매이력 텍스트}
```

**번역**

```text
천천히 숨을 고르고 지시에 따라 단계별로 작업하세요.
이제 Funnel E-commerce 데이터셋을 분석합니다.
이 데이터셋은 온라인 리테일러에서 각 유저의 구매 거래와 구매 상품이 기록된 것입니다.
약 100명의 유저 구매 정보를 제공합니다. 각 유저 데이터는 ID로 그룹화되어 자연어로 설명됩니다.

목표: 구매 패턴을 기반으로 대표적이고 정확한 유저 페르소나 20개를 생성하세요.
두 가지 목표를 최적화하세요:

• 높은 커버리지: 가능한 많은 유저에게 하나 이상의 페르소나를 부여할 수 있어야 합니다.
• 높은 정확도: 각 페르소나는 명확하고 모호하지 않은 정의를 가져야 합니다.

각 페르소나에 이름과 한 문장 정의를 제공하세요.
정확히 20개의 페르소나를 아래 형식으로 출력하세요:
1. [페르소나명] - [정의]
...
20. [페르소나명] - [정의]
```

---

### Step 2 — 후보 정제

**시스템 프롬프트** (Step 3과 동일)

```text
You are an assistant skilled at reading, observing and summarizing, capable of
finding similar or repeated descriptions of user personas, and good at finding
the most representative ones.
```

**번역**

```text
당신은 읽기, 관찰, 요약에 능숙하고, 유사하거나 중복된 유저 페르소나 설명을 찾아내며
가장 대표적인 것을 선별하는 데 뛰어난 어시스턴트입니다.
```

---

**유저 프롬프트**

```text
Take a deep breath and work according to the instructions step by step.
I will give you 5 persona sets (each containing 20 personas), totaling 100 personas.
Select the 20 most representative personas from these 100.
If a set contains fewer than 20 personas or irrelevant information, ignore that set.
You may find similar or duplicate personas — merge or pick the most representative.
Do NOT use occurrence count as a criterion; judge solely by the persona descriptions.

Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

Here are the five persona sets:
{5개 페르소나 세트}
```

**번역**

```text
천천히 숨을 고르고 지시에 따라 단계별로 작업하세요.
20개씩 구성된 5개의 페르소나 세트, 총 100개의 페르소나를 제공합니다.
이 100개 중 가장 대표적인 20개를 선택하세요.
세트에 20개 미만이 있거나 관련 없는 정보가 있으면 무시하세요.
유사하거나 중복된 페르소나는 병합하거나 가장 대표적인 것을 선택하세요.
등장 횟수는 기준으로 삼지 마세요. 오직 페르소나 설명만으로 판단하세요.
```

---

### Step 3 — 최종 확정

**유저 프롬프트**

```text
Take a deep breath and work according to the instructions step by step.
I will give you 8 persona sets (each containing 20 personas), totaling 160 personas.
Select the 20 most representative and distinct final personas from these 160.
Prefer personas that appear most frequently across sets and cover the broadest user base.

Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

Here are the eight persona sets:
{8개 페르소나 세트}
```

**번역**

```text
20개씩 구성된 8개의 페르소나 세트, 총 160개의 페르소나를 제공합니다.
이 중 가장 대표적이고 구분되는 최종 20개를 선택하세요.
여러 세트에서 가장 자주 등장하고 가장 넓은 유저 기반을 커버하는 페르소나를 우선시하세요.
```

---

## 2. 페르소나 라벨링 (Labeling)

### 유저 라벨링

**시스템 프롬프트**

```text
Now you are an intelligent e-commerce domain assistant.
You are skilled at summarizing, and capable of assigning high-level
consumer personas based on a user's purchase behavior.
```

**번역**

```text
당신은 지능적인 이커머스 도메인 어시스턴트입니다.
요약에 능숙하고, 유저의 구매 행동을 기반으로 고수준 소비자 페르소나를 부여할 수 있습니다.
```

---

**유저 프롬프트**

```text
Take a deep breath and work according to the instructions step by step.
Your goal is to identify users' shopping behaviors based on products they have bought
and label them with a given set of personas. Select at least one persona, at most 5 personas
from the given list. Make sure each assignment has strong evidence in their purchase transactions.

Select only exact persona names from the given list.
Do not rename, paraphrase, abbreviate, or create new personas.

Only output valid JSON. No markdown. No explanation.

Output format:
{ "user_number": ["Persona1", "Persona2"] }

Example:
{ "1001": ["Tech & Gadget Shopper", "Frequent Repurchaser"] }

If no persona fits, label as Unrepresentable:
{ "9999": ["Unrepresentable"] }

Here is the persona list you should choose from:
{페르소나 목록}

Remember: the user number must match exactly from the transaction data.

Here is user {uid}'s transaction data:
{describe_user() 출력}

Select only from the given persona list.
Only output valid JSON. No markdown. No explanation.
```

**번역**

```text
천천히 숨을 고르고 지시에 따라 단계별로 작업하세요.
목표: 유저가 구매한 상품을 기반으로 쇼핑 행동을 파악하고 주어진 페르소나 세트에서 라벨을 부여하세요.
주어진 목록에서 최소 1개, 최대 5개의 페르소나를 선택하세요.
각 부여에는 구매 거래에서 강력한 근거가 있어야 합니다.

주어진 목록에서 정확한 페르소나 이름만 선택하세요.
이름을 바꾸거나, 바꿔 말하거나, 줄이거나, 새로운 페르소나를 만들지 마세요.

유효한 JSON만 출력하세요. 마크다운 없음. 설명 없음.
```

---

### 상품 라벨링

**시스템 프롬프트**

```text
Now you are an intelligent e-commerce domain assistant.
You have a good understanding of various customer personas,
and are skilled at connecting products with personas most likely to be interested in them.
```

**번역**

```text
당신은 지능적인 이커머스 도메인 어시스턴트입니다.
다양한 고객 페르소나를 잘 이해하고 있으며,
상품과 관심을 가질 가능성이 높은 페르소나를 연결하는 데 뛰어납니다.
```

---

**유저 프롬프트**

```text
Take a deep breath and work according to the instructions step by step.

I will provide a product name and a set of customer personas.
Your task is to connect the product with a subset of personas that would likely purchase it.

Principles:
1. Understand each persona strictly by its definition.
2. Use real-world knowledge to understand the product's attributes from its name.
3. Only connect personas with strong, well-substantiated relevance to the product.

Select only exact persona names from the given list.
Do not rename, paraphrase, abbreviate, or create new personas.

Only output valid JSON. No markdown. No explanation.

Personas:
{페르소나 목록}

Output format: a JSON dict mapping the exact product name to a list of 1–5 relevant persona names.
Example: {"SSD MediumBlue 149": ["Tech & Gadget Shopper", "Single-Category Specialist"]}

Product name: {상품명}
```

**번역**

```text
상품명과 고객 페르소나 세트를 제공합니다.
해당 상품을 구매할 가능성이 높은 페르소나 부분 집합을 연결하는 것이 목표입니다.

원칙:
1. 각 페르소나를 정의에 따라 엄격하게 이해하세요.
2. 실세계 지식을 사용해 상품명에서 속성을 파악하세요.
3. 상품과 강력하고 충분히 뒷받침된 관련성이 있는 페르소나만 연결하세요.

유효한 JSON만 출력하세요. 마크다운 없음. 설명 없음.
```

---

## 3. 유저 텍스트 포맷 (`describe_user()`)

### 기본 형식 (M1 — 구매 상품명 + 횟수)

```text
The user 12345 has totally purchased 5 unique products.
Each product name is followed by its purchased times:
PRODUCT_A, 3 times; PRODUCT_B, 1 time.
```

### 확장 형식 (M2 — 구매 상품 + 행동 컨텍스트)

`customer_features`가 전달된 경우 아래 블록이 추가된다.

```text
The user 12345 has totally purchased 5 unique products.
Each product name is followed by its purchased times:
PRODUCT_A, 3 times; PRODUCT_B, 1 time.

Behavioral context:
- Sessions: 15
- Page views: 45, Add-to-carts: 8 (ATC/PV: 17.8%)
- Orders: 3, Total spend: $234.50, Average order value: $78.17
- Last session: 32 days ago, Last order: 45 days ago
- Most viewed category: Books, Most purchased category: Beauty
```

### 결측값 처리 규칙

| 항목 | 결측 시 표시 |
|------|------------|
| `recency_order_days` | `N/A` |
| `avg_order_value` | `N/A` |
| `top_purchase_category` | `N/A` |
| `page_view_count = 0` | ATC/PV → `N/A` |

---

## 3. Segment Naming (v2 — Issue #17)

> **관련 파일**: `src/persona/segment_naming.py`
> **입력**: `data/processed/segment_summary_all_customers.csv` (Issue #16 산출물)
> **모델**: Upstage Solar Pro (`solar-pro`), `temperature=0` (재현성 확보를 위해 고정 — 10회 반복 실행 안정성 검증 결과는 `reports/[ML]_Segment_Naming_v2_LLM_라벨링_2026-07-03.md` 참고)
> **실험 버전 관리**: `experiments/segment_naming_v2/` (run별 산출물 + `CHOICES.md` 채택 기록), `--promote`로 canonical 파일 반영

v1과의 핵심 차이: LLM은 persona taxonomy를 만들거나 개별 고객을 분류하지 않는다.
KMeans가 이미 확정한 segment_id별 집계 통계만 보고 이름/설명을 붙이는
라벨러 역할로 제한된다. 개별 고객 row, raw 상품명, raw log는 입력에 넣지 않는다.

**시스템 프롬프트**

```text
You are naming customer behavior segments for an ecommerce analytics system.
You act strictly as a labeler: you translate already-computed segment
statistics into a human-readable name and description. You are not analyzing
individual customers and you are not inventing a persona taxonomy.
```

**번역**

```text
당신은 이커머스 분석 시스템의 고객 행동 세그먼트에 이름을 붙이는 역할입니다.
당신은 엄격히 라벨러로서, 이미 계산된 세그먼트 통계를 사람이 읽을 수 있는
이름과 설명으로 번역합니다. 개별 고객을 분석하거나 페르소나 체계를
새로 만들어내지 않습니다.
```

---

**유저 프롬프트** (`build_user_prompt()` 출력)

```text
Take a deep breath and work according to the instructions step by step.

{format_segment_summary() 출력 — Segment N summary 통계 블록}

Task:
Create a concise segment name and description using only the evidence above.

Rules:
- Use only the statistics given above. Do not use raw product names, individual
customer records, or any data not shown in this summary.
- Do not infer income, age, occupation, family status, or lifestyle. These are not
observable in the data.
- Prefer observable behavior-based phrasing (e.g. High-Intent, Repeat,
Non-Purchasing, Browsing-Only) over demographic/lifestyle phrasing (e.g. Young,
Urban, Affluent, Family, Professional).
- This segment was produced by KMeans clustering with generally low silhouette
scores, so treat it as a behavior-based approximation, not a sharply separated
natural group. Avoid overconfident or marketing-style labels.
- Every item in "evidence" must restate a specific field/value from the summary
above. Do not add evidence that isn't in the summary.
- If a field above is "none", do not claim behavior about it.
- Return valid JSON only. No markdown. No explanation.

Output JSON schema:
{
  "segment_id": <integer, must equal N>,
  "segment_name": "<string>",
  "description": "<string>",
  "evidence": ["<string>", ...],
  "cautions": ["<string>", ...]
}
```

**번역**

```text
천천히 숨을 고르고 지시에 따라 단계별로 작업하세요.

{세그먼트 통계 블록}

목표: 위 근거만 사용해 간결한 세그먼트 이름과 설명을 작성하세요.

규칙:
- 위에 주어진 통계만 사용하세요. raw 상품명, 개별 고객 기록, 요약에 없는
  데이터는 사용하지 마세요.
- 소득, 연령, 직업, 가족 상태, 라이프스타일을 추론하지 마세요. 데이터로
  관측되지 않습니다.
- 데모그래픽/라이프스타일 표현(Young, Urban, Affluent, Family, Professional)보다
  관측 가능한 행동 기반 표현(High-Intent, Repeat, Non-Purchasing, Browsing-Only)을
  우선하세요.
- 이 세그먼트는 실루엣 스코어가 대체로 낮은 KMeans clustering 결과이므로,
  명확히 분리된 자연 군집이 아니라 행동 기반 근사치로 취급하세요. 과신하는
  라벨이나 마케팅 문구를 피하세요.
- "evidence"의 각 항목은 위 요약의 구체적 필드/값을 그대로 반영해야 합니다.
  요약에 없는 근거를 추가하지 마세요.
- 위 필드가 "none"이면 그 항목에 대해 행동을 단정하지 마세요.
- 유효한 JSON만 출력하세요. 마크다운 없음. 설명 없음.
```

---

**Segment summary → 텍스트 변환 규칙** (`format_segment_summary()`)

| 항목 | 규칙 |
|------|------|
| 비율 값(`*_ratio`, `*_rate`) | 퍼센트 소수점 1자리, 결측 시 `none` |
| 평균 값(`avg_*`) | 소수점 2자리(recency는 1자리), 결측 시 `none` |
| 카테고리 분포(`top_*_categories`) | 이미 `"Books: 20.2%; Beauty: 19.0%"` 형태 문자열, 결측/빈 문자열이면 `none` |

이 규칙은 v1의 `describe_user()` 결측값 처리 관례(§3)를 그대로 따른다.

---

## 4. 주의사항

| 문제 | 방지 방법 |
|------|----------|
| JSON 파싱 실패 | 프롬프트에 `Only output valid JSON. No markdown. No explanation.` 명시 |
| 페르소나명 변형 출력 | `Select only exact persona names from the given list. Do not rename, paraphrase, abbreviate, or create new personas.` 명시 |
| 정성어 과다 사용 | 원시 수치와 비율을 입력, LLM이 해석하도록 유도 |
