# Data Catalog — Raw 데이터

**데이터 수집 기간**: 2020-01-01 ~ 2025-10-31  
**데이터 성격**: Synthetic dataset (실제 서비스 로그 아님)  
**위치**: `data/raw/`

---

## 테이블 목록

| 테이블 | 파일 | 행 수 | 컬럼 수 | 설명 |
|--------|------|-------|---------|------|
| `customers` | `customers.csv` | 20,000 | 7 | 고객 기본 정보 |
| `sessions` | `sessions.csv` | 120,000 | 6 | 세션 단위 방문 기록 |
| `events` | `events.csv` | 760,958 | 10 | 세션 내 이벤트 로그 |
| `orders` | `orders.csv` | 33,580 | 10 | 주문 헤더 (결제 단위) |
| `order_items` | `order_items.csv` | 59,163 | 5 | 주문 상세 (상품 단위) |
| `products` | `products.csv` | 1,197 | 6 | 상품 정보 |
| `reviews` | `reviews.csv` | 10,780 | 6 | 상품 리뷰 |

---

## 테이블 상세

### customers

고객 마스터 테이블.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `customer_id` | int | 없음 | PK |
| `name` | str | 없음 | 고객명 (합성 데이터) |
| `email` | str | 없음 | 이메일 (합성 데이터) |
| `country` | str | 없음 | 국가 코드 (ISO 2자리) |
| `age` | int | 없음 | 나이 (18 ~ 75) |
| `signup_date` | str | 없음 | 가입일 (2020-01-01 ~ 2025-10-31) |
| `marketing_opt_in` | bool | 없음 | 마케팅 수신 동의 여부 |

**주요 통계**
- 국가: 17개국, 상위 5개 — US(3,648), IN(1,589), GB(1,585), BR(1,421), DE(1,397)
- 마케팅 동의: True 55.6% / False 44.4%

**US cohort 정의**  
US 고객 분석은 반드시 `customers.country == 'US'` 기준으로 cohort를 먼저 정의한 뒤, 해당 고객들의 세션/이벤트/주문을 따라간다. `sessions.country` 또는 `orders.country`로 필터링하지 않는다.

---

### sessions

고객별 세션 방문 기록. 고객 1명당 평균 6.0회 세션.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `session_id` | int | 없음 | PK |
| `customer_id` | int | 없음 | FK → customers |
| `start_time` | str | 없음 | 세션 시작 시각 (ISO 8601) |
| `device` | str | 없음 | 접속 기기 |
| `source` | str | 없음 | 유입 채널 |
| `country` | str | 없음 | 세션 접속 국가 |

**주요 통계**
- device: mobile(54.9%), desktop(37.9%), tablet(7.1%)
- source: organic(34.0%), direct(24.9%), paid(12.1%), social(12.0%), email(9.1%), referral(8.0%)

---

### events

세션 내 행동 이벤트 로그.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `event_id` | int | 없음 | PK |
| `session_id` | int | 없음 | FK → sessions |
| `timestamp` | str | 없음 | 이벤트 발생 시각 (ISO 8601) |
| `event_type` | str | 없음 | 이벤트 유형 |
| `product_id` | float | 78,489 | FK → products (`checkout`, `purchase`는 null. `checkout`은 복원 가능 — 아래 참고) |
| `qty` | float | 617,832 | 수량 (`add_to_cart`만 존재) |
| `cart_size` | float | 716,049 | 장바구니 내 상품 수 (`checkout`만 존재) |
| `payment` | str | 727,378 | 결제 수단 (`purchase`만 존재) |
| `discount_pct` | float | 727,378 | 할인율 (`purchase`만 존재) |
| `amount_usd` | float | 727,378 | 결제 금액 (`purchase`만 존재) |

**주요 통계**
- event_type: page_view(70.9%), add_to_cart(18.8%), checkout(5.9%), purchase(4.4%)
- `checkout`, `purchase` 이벤트는 `product_id`가 null

**`checkout` product_id 복원 (Issue #20 검증, `notebooks/EDA.ipynb`)**  
`remove_from_cart` 이벤트가 존재하지 않아, 같은 `session_id`의 `add_to_cart` 상품 목록이 checkout 시점까지 그대로 유지된다. checkout이 있는 전체 세션(44,909개)에서 `add_to_cart` qty 합과 `cart_size`가 100% 일치함을 확인했다 — `add_to_cart` 로그로 checkout 상품을 복원할 수 있다(`src/datasets/make_als_mart.py` 적용 사례). 단, checkout 이후 타임스탬프로 찍히는 `add_to_cart`가 일부 세션(306개)에 존재하므로 시간순 필터링 없이 세션 전체의 `add_to_cart`를 사용해야 한다.

`purchase` 이벤트는 별도 검증된 복원 방법이 없다 — 구매 상품 분석에는 `orders`/`order_items`(둘 다 `product_id` 결측 없음)를 사용한다.

---

### orders

주문 헤더 테이블. 결제 1건 = 1행.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `order_id` | int | 없음 | PK |
| `customer_id` | int | 없음 | FK → customers |
| `order_time` | str | 없음 | 주문 시각 (ISO 8601) |
| `payment_method` | str | 없음 | 결제 수단 |
| `discount_pct` | int | 없음 | 할인율 (0, 5, 10, 15, 20) |
| `subtotal_usd` | float | 없음 | 할인 전 금액 |
| `total_usd` | float | 없음 | 최종 결제 금액 |
| `country` | str | 없음 | 주문 국가 |
| `device` | str | 없음 | 주문 기기 |
| `source` | str | 없음 | 유입 채널 |

**주요 통계**
- payment_method: card(69.8%), paypal(15.0%), wallet(10.0%), cod(5.1%)
- total_usd: $2.80 ~ $2,984.58

---

### order_items

주문 상세 테이블. 주문 1건에 여러 상품이 포함될 수 있어 `orders`와 1:N 관계.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `order_id` | int | 없음 | FK → orders |
| `product_id` | int | 없음 | FK → products |
| `unit_price_usd` | float | 없음 | 상품 단가 |
| `quantity` | int | 없음 | 구매 수량 |
| `line_total_usd` | float | 없음 | 상품별 소계 (unit_price × quantity) |

**주의**: `total_usd`(주문 합계)를 집계할 때는 반드시 `orders` 테이블을 사용하거나 `order_id` 기준으로 dedup 후 사용해야 한다. `order_items`에서 `total_usd`를 그대로 sum하면 상품 수만큼 중복 합산된다.

---

### products

상품 마스터 테이블.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `product_id` | int | 없음 | PK |
| `category` | str | 없음 | 상품 카테고리 |
| `name` | str | 없음 | 상품명 |
| `price_usd` | float | 없음 | 판매가 |
| `cost_usd` | float | 없음 | 원가 |
| `margin_usd` | float | 없음 | 마진 (price - cost) |

**주요 통계**
- category: Electronics, Home & Kitchen, Beauty, Sports, Fashion, Books, Toys — 각 171개로 균등 분포
- price_usd: $3.50 ~ $596.62

---

### reviews

상품 리뷰 테이블. 전체 주문의 18.2%에만 리뷰 존재.

| 컬럼 | 타입 | 결측 | 설명 |
|------|------|------|------|
| `review_id` | int | 없음 | PK |
| `order_id` | int | 없음 | FK → orders |
| `product_id` | int | 없음 | FK → products |
| `rating` | int | 없음 | 평점 (1 ~ 5) |
| `review_text` | str | 없음 | 리뷰 텍스트 |
| `review_time` | str | 없음 | 리뷰 작성 시각 |

**주요 통계**
- rating 분포: 1점(3.9%), 2점(7.0%), 3점(18.4%), 4점(33.1%), 5점(37.6%)
- 리뷰 있는 주문: 6,111건 / 33,580건 (18.2%)

---

## 조인 구조

```
customers (20,000)
  └─ sessions (120,000)  customer_id
       └─ events (760,958)  session_id
            └─ products (1,197)  product_id  [page_view, add_to_cart만]

customers (20,000)
  └─ orders (33,580)  customer_id
       ├─ order_items (59,163)  order_id
       │    └─ products (1,197)  product_id
       └─ reviews (10,780)  order_id
            └─ products (1,197)  product_id
```

---

## 주의사항

- **합성 데이터**: 실제 시장/고객/상품에 대한 결론으로 일반화하지 않는다.
- **`checkout`·`purchase` 이벤트**: `product_id`가 null. `checkout`은 같은 세션의 `add_to_cart` 로그로 복원 가능(Issue #20, 위 `events` 절 참고). `purchase` 상품 분석은 `orders`/`order_items`를 사용한다.
- **`total_usd` 중복 주의**: `order_items`와 조인 후 `total_usd`를 집계할 때 반드시 `order_id` 기준 dedup 필요.
- **US cohort 정의**: US 분석은 `customers.country == 'US'` 기준이며, `sessions.country`나 `orders.country`로 필터링하지 않는다.
