# LightGCN_tri 성능 개선 계획

## 배경

첫 전체 학습 결과(`reports/[ML]_LightGCN_tri_전체학습_및_ALS_비교평가_2026-07-08.md`), 하이퍼파라미터를 하나도 안 건드린 기본값 상태에서 ALS 대비 HR@20이 절반 수준(0.0294 vs 0.0608)이었다. 이 문서는 "그다음에 뭘 조정해볼 수 있는지"를 정리한다.

**중요한 선행 조치**: 처음엔 "학습이 애초에 되고 있는지"조차 판단할 수 없는 상태였다(loss 로깅이 꺼져 있었고, 평가가 매 epoch 다른 무작위 512명 표본이라 곡선이 순수 노이즈였음). 이 두 가지는 이미 고쳤다(`train_model.py`/`test_model.py` 수정, #30) — 그 결과 loss가 실제로 꾸준히 감소하는 게 확인됐다(3 epoch 스모크 테스트 기준 4332→4230→4007). 즉 **최적화 자체는 정상 작동**하고, 아래는 "얼마나 잘/빠르게 좋은 지점에 도달하는가"를 개선하는 항목들이다.

## ⚠️ bipartite(#34/#35)와 비교 조건 통일 — 튜닝 전에 반드시 확인

tri를 튜닝하는 동안 누군가 bipartite도 별도로 실험 중이라면, **아래 항목이 서로 다르면 #34의 tri vs bipartite 비교 자체가 무효해진다.** "페르소나 효과"를 보려는 건데, 조건까지 다르면 그 차이가 페르소나 때문인지 조건 차이 때문인지 구분이 안 된다.

| 항목 | 반드시 통일해야 함 | 비고 |
|---|---|---|
| `epoch` | 같은 값 (또는 같은 기준으로 early stopping) | 이 문서 #1 항목에서 값이 바뀌면 bipartite도 같이 바꿔야 함 |
| u2t event_type 조합 | 같은 조합 | 이 문서 #2 항목 — tri에서 정한 조합을 bipartite 데이터 생성(#35)에도 동일 적용 |
| `lr`, `lamda`, `layer`, `emb_dim` | 같은 값 | 이 문서 #3 항목 — tri만 튜닝하고 bipartite는 기본값 쓰면 불공정 비교 |
| 평가 방식 | 같은 고정 유저 집합, 같은 K 리스트 | **이건 이미 안전함** — `train_model.py`/`test_model.py`가 그래프 구조와 무관한 공용 코드라, 이번에 고친 loss 로깅/고정 평가셋이 bipartite 실행에도 그대로 적용됨 |
| `split_date` | 2025-08-01 (ALS와 동일, #29에서 이미 고정) | tri/bipartite/ALS 셋 다 동일 |

**tri 쪽에서 최종 조합(epoch 수, event_type, 하이퍼파라미터)이 정해지면, 그 값을 bipartite 실행에도 그대로 넘겨야 한다.** bipartite를 실험 중인 사람이 있다면 이 표를 공유해서 조건을 맞춰야 한다 — 이슈 #34에 코멘트로 남겨두는 걸 추천한다.

## 개선 후보 (우선순위 순)

### 1. 학습 epoch 수 — 가장 먼저 확인할 것

loss가 300 epoch 시점에도 계속 감소 중이었을 가능성이 있다(첫 실행 땐 loss를 안 봐서 확인 못 했음). **지금 다시 돌리고 있는 300 epoch 실행의 loss 곡선을 보고, 여전히 뚜렷하게 감소 중이면 epoch을 늘려야 한다** (500~1000 등). ALS는 `iterations=20`으로 훨씬 적게 도는 다른 종류의 최적화(ALS closed-form)라 직접 비교는 안 되지만, LightGCN(SGD 계열)은 원래 더 많은 epoch이 필요한 경우가 많다.

- **비용**: epoch당 약 15~20초 → 500 epoch면 약 2~2.5시간
- **확인 방법**: `logs/LightGCN/run_lightgcn_train_log.xlsx`의 `Loss` 시트에서 마지막 구간 기울기 확인

### 2. u2t event_type 조합 — #29에서 이미 준비해둔 실험

`src/datasets/make_lgcn_graph.py`의 `DEFAULT_U2T_EVENT_TYPES = ["page_view", "add_to_cart", "purchase"]`. 지금은 셋 다 동일 가중치(1) 무방향 엣지로 합쳐서 쓰는데, `page_view`가 너무 많이 섞이면 "관심 있어서 봤다"보다 "그냥 스쳐 지나갔다"는 노이즈가 커질 수 있다.

시도해볼 조합:
| 조합 | 가설 |
|---|---|
| `purchase`만 | 가장 강한 신호, 그래프가 희소해짐(구매 안 한 유저는 고립) |
| `add_to_cart + purchase` | page_view 노이즈 제거, 관심도 있는 행동만 |
| 전부(현재) | 밀도는 높지만 신호 대 잡음비 낮을 수 있음 |

- **비용**: `make_lgcn_graph.py --event-types ...` 재실행(#29 인프라 재사용, 분 단위) + 학습 재실행(시간 단위)
- **평가**: `evaluate_lightgcn.py`로 HR@20 비교

### 3. 하이퍼파라미터 튜닝

현재 값은 전부 `parse.py`의 원 논문 기본값(다른 데이터셋 대상)이다.

| 파라미터 | 현재 값 | 튜닝 방향 |
|---|---|---|
| `lr` | 0.005 | ALS도 그리드서치로 별도 값 찾았음(`notebooks/20260703_ML_als_hyperparameter_tuning.ipynb` 참고) — LightGCN도 동일 패턴 필요 |
| `lamda` (L2 정규화) | 0.02 | ALS의 `regularization=10.0`과 스케일이 다른 지표라 직접 비교는 안 되지만, 과적합/과소적합 여부를 보고 조정 |
| `layer` | 2 | LightGCN 논문은 보통 2~4 레이어 권장. 레이어가 너무 많으면 over-smoothing(모든 임베딩이 비슷해짐) 위험 |
| `emb_dim` | 128 (`pred_dim` 기본값) | ALS의 `factors=16`과 비교하면 상당히 큼 — 유저 20,000명·상품 1,197개 규모에 128차원이 과한지 확인 필요 |

- **비용**: 조합마다 학습 1회(시간 단위) — ALS 그리드서치처럼 노트북으로 여러 조합을 자동화하면 효율적
- **주의**: #1(epoch 수)과 #2(event_type)를 먼저 정하고 나서 하이퍼파라미터를 튜닝하는 게 순서상 맞다 — 안 그러면 "잘못된 조건에서 최적화된 하이퍼파라미터"를 얻을 위험이 있다

### 4. t2p lift 임계값 / 최소 구매 건수 (구조적 선택, 우선순위 낮음)

`LIFT_THRESHOLD = 1.15`, `MIN_PURCHASE_COUNT = 5` — #29 리포트에도 "이론적 근거보다는 합리적인 시작값"이라고 명시돼 있다. 세그먼트-상품 연결이 너무 적거나(1,197개 중 1,143개만 연결) 너무 헐거우면 페르소나 신호가 약해질 수 있다. 다만 이건 tri 그래프 구조 자체를 바꾸는 거라 영향 범위가 크고, 위 1~3번보다 우선순위를 낮게 둔다.

## 실행 순서 제안

```text
1. 지금 돌고 있는 300 epoch 결과의 loss 곡선 확인
   → 계속 감소 중이면: epoch 늘려서 재실행
   → 수렴했으면: 다음 단계로
2. u2t event_type 조합 3가지 비교 (구매만 / 장바구니+구매 / 전부)
   → HR@20 기준 최선 조합 선정
3. 그 조합으로 하이퍼파라미터 튜닝 (lr/lamda/layer/emb_dim)
4. (여유 있으면) lift threshold/min purchase count 재검토
```

## 관련 문서

- 현재 baseline 결과: `reports/[ML]_LightGCN_tri_전체학습_및_ALS_비교평가_2026-07-08.md`
- ALS 튜닝 사례(참고 패턴): `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb`, `reports/[ML]_#7_ALS_성능개선_및_Threshold_검증_2026-07-03.md`
- t2p lift 설계 배경: `docs/WHY_SEGMENTS.md`, `reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`
