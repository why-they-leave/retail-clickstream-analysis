# Online Retail Clickstream Analysis

가상 온라인 리테일 클릭스트림 데이터를 활용해 고객 행동과 전환 패턴을 분석하는 프로젝트입니다.

이 프로젝트는 Kaggle의 synthetic online retail clickstream dataset을 바탕으로 온라인 쇼핑 행동, 구매 퍼널, 전환 행동 피처 설계, 테이블 조인 구조를 검증합니다. 데이터가 실제 서비스 로그가 아닌 합성 데이터이므로, 분석 결과를 실제 비즈니스 인사이트로 단정하기보다 클릭스트림 기반 퍼널 분석 방법론과 재현 가능한 분석 파이프라인을 점검하는 데 목적을 둡니다.

---

## 분석 목표

- 고객의 세션, 이벤트, 주문 데이터를 연결해 구매 전환 흐름을 복원합니다.
- 클릭스트림 이벤트에서 퍼널 단계별 이탈과 전환 패턴을 분석합니다.
- 주문, 주문 상세, 상품 테이블을 조인해 구매 상품 정보를 복원합니다.
- 전환 행동을 설명할 수 있는 세션/고객 단위 피처를 설계합니다.
- 합성 데이터 환경에서 재현 가능한 분석 파이프라인을 구성합니다.

---

## 데이터셋

Source:

- Kaggle: `wafaaelhusseini/e-commerce-transactions-clickstream`

범위:

| 항목 | 값 |
|------|----|
| 수집 기간 | `2020-01-01` ~ `2025-10-31` |
| 전체 고객 수 | 20,000명 |
| US 고객 수 | 3,648명 |
| 전체 이벤트 수 | 760,958건 |
| US 이벤트 수 | 138,891건 |
| 전체 주문 수 | 33,580건 |
| US 주문 수 | 6,138건 |

원본 데이터는 `data/raw/`에 두고, 중간 처리 결과는 `data/interim/`, 분석 또는 모델 입력용 최종 데이터는 `data/processed/`에 저장합니다. 원본 데이터 파일은 Git으로 추적하지 않습니다.

---

## 주요 테이블

분석에 사용하는 주요 테이블은 다음 7개입니다.

| 테이블 | 설명 |
|--------|------|
| `customers` | 고객 프로필 |
| `products` | 상품 정보 |
| `sessions` | 세션 메타데이터 |
| `events` | 클릭스트림 행동 로그 |
| `orders` | 주문 정보 |
| `order_items` | 주문 상세 상품 정보 |
| `reviews` | 상품 리뷰 |

### ERD (Entity-Relationship Diagram)

![Schema ERD](reports/schema_erd.png)

---

## 핵심 조인

### 구매 상품명 복원

고객이 구매한 상품명과 카테고리는 `orders`, `order_items`, `products` 세 테이블을 조인해 복원합니다.

```text
orders(order_id, customer_id)
  -> order_items(order_id, product_id)
  -> products(product_id, name, category)
```

예상 산출 컬럼:

- `customer_id`
- `order_id`
- `product_id`
- `name`
- `category`
- 주문 또는 주문 상세 단위의 수량, 가격, 금액 관련 컬럼

### 클릭스트림과 전환 연결

세션 단위 행동과 구매 전환은 고객 및 세션 식별자를 기준으로 연결합니다.

```text
customers
  -> sessions
  -> events
  -> orders
```

분석 시점에 따라 고객 단위, 세션 단위, 이벤트 단위 중 하나를 기준 grain으로 정하고 조인 중복과 집계 기준을 명확히 관리합니다.

---

## 분석 주제

### 퍼널 분석

- 이벤트 타입별 고객 이동 경로
- 세션 시작 후 구매까지의 단계별 전환율
- 장바구니, 상품 조회, 구매 등 주요 행동 간 전환
- 국가 또는 고객 세그먼트별 퍼널 차이

### 전환 행동 피처

- 세션 내 이벤트 수
- 상품 조회 수와 고유 상품 수
- 카테고리 다양성
- 장바구니 또는 checkout 관련 이벤트 발생 여부
- 첫 이벤트부터 구매까지의 시간
- 고객의 과거 주문 수와 누적 구매 금액

### 주문 상품 분석

- 구매 상품 카테고리 분포
- 고객별 구매 상품 다양성
- 주문 금액과 이벤트 행동의 관계
- 리뷰 데이터와 구매 상품 정보의 연결 가능성

---

## 프로젝트 구조

```text
online-retail-clickstream-analysis/
├── README.md                     # 프로젝트 설명과 사용 방법
├── CHANGELOG.md                  # 변경 이력
├── CLAUDE.md                     # Claude Code용 프로젝트 지침
├── pyproject.toml                # 패키지 메타데이터와 도구 설정
├── uv.lock                       # uv 잠금 파일
├── requirements.txt              # 핵심 의존성 목록
│
├── configs/
│   ├── base.yaml                 # 공통 설정
│   ├── dev.yaml                  # 개발 환경 설정
│   └── prod.yaml                 # 제출/운영 환경 설정
│
├── data/
│   ├── raw/                      # 원본 데이터, git 추적 제외
│   ├── interim/                  # 중간 처리 데이터
│   └── processed/                # 분석/모델 입력용 최종 데이터
│
├── notebooks/                    # 탐색 분석과 실험 노트북
├── reports/                      # 보고서, 그림, 표, 대시보드 산출물
│
├── src/
│   ├── features/                 # 피처 생성 코드
│   ├── modeling/                 # 모델 학습 코드
│   ├── evaluation/               # 평가 지표와 검증 코드
│   └── visualization/            # 시각화 코드
│
└── tests/                        # 단위 테스트
```

---

## 개발 환경

이 프로젝트는 Python 3.11 이상과 `uv` 기반 의존성 관리를 사용합니다.

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

`pip` 환경에서는 아래처럼 실행할 수 있습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

Windows PowerShell에서는 가상환경을 다음처럼 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

---

## 품질 확인

커밋하거나 PR을 열기 전에 아래 명령을 실행합니다.

```bash
uv run --extra dev ruff check src/ tests/
uv run --extra dev ruff format --check src/ tests/
uv run --extra dev pytest tests/ -v
```

---

## 분석 원칙

### 합성 데이터 해석

- 본 데이터셋은 synthetic dataset이므로 실제 시장, 고객, 상품에 대한 결론으로 일반화하지 않습니다.
- 수치 결과보다 분석 절차, 조인 구조, 피처 설계, 재현성 검증에 초점을 둡니다.
- 데이터 생성 방식의 편향 가능성과 실제 서비스 로그와의 차이를 보고서에 명시합니다.

### 재현성

- random seed를 명시합니다.
- 데이터 필터링 기준과 train/validation/test 분할 기준을 코드나 문서에 남깁니다.
- 원본 데이터는 수정하지 않고 처리 결과만 `data/interim/` 또는 `data/processed/`에 저장합니다.

### 데이터 누수 방지

- 학습 목적의 피처 생성은 train/validation/test 분리 후 수행합니다.
- 인코더, 스케일러, imputing 파라미터는 학습 데이터에만 fit합니다.
- 시계열 rolling/lag 피처는 미래 값을 참조하지 않도록 `shift(1)` 이후 계산합니다.

### 설명 가능한 결과

- 모델 성능뿐 아니라 데이터 가정, 한계, 실패 사례를 함께 기록합니다.
- 복잡한 모델을 쓰기 전에 단순한 baseline을 먼저 만듭니다.
- 주요 판단은 노트북, 이슈, PR 설명, 보고서 중 한 곳에 남깁니다.

---

## 변경 이력

변경 이력은 별도 [`CHANGELOG.md`](CHANGELOG.md) 파일로 관리합니다.
