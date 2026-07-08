# LightGCN_tri 성능 개선 계획

> **상태 (2026-07-08 갱신)**: 아래 계획했던 항목(epoch, event_type, 하이퍼파라미터, layer/optimizer/batch, negative sampling)은 전부 실험 완료됐다. 최종 확정 설정과 근거는 `reports/[ML]_LightGCN_tri_전체학습_및_ALS_비교평가_2026-07-08.md`(3~5차 실행 섹션)를 참고. 이 문서는 배경 설명과 실험 과정 기록으로 남겨두고, 최신 수치는 항상 리포트를 우선한다.

## 배경

첫 전체 학습 결과, 하이퍼파라미터를 하나도 안 건드린 기본값 상태에서 ALS 대비 HR@20이 절반 수준(0.0294 vs 0.0608)이었다. 이 문서는 "그다음에 뭘 조정해볼 수 있는지"를 정리했고, 아래 순서대로 전부 실험했다.

**선행 조치(완료)**: 학습 진단(loss 로깅, 고정 평가셋)을 2차에서 고쳤다. 또한 `#30` PR 리뷰 중 CodeRabbit이 레이어 결합 가중치 실버그(표준 uniform이 아니라 레이어0에 편중된 decay 방식으로 동작 중)를 발견해 3~4차 결과는 전부 이 버그 상태에서 나온 것이었다 — 5차(#37)에서 고치고 재검증했다.

## ⚠️ bipartite(#34/#35)와 비교 조건 통일 — 최종 확정값

tri 튜닝이 끝났다. bipartite 실험 시 아래 값을 그대로 써야 `#34`의 tri vs bipartite 비교가 유효하다.

| 항목 | 확정값 | 비고 |
|---|---|---|
| `epoch` | 300 | |
| u2t event_type 조합 | 구매만 (`purchase`) | `make_lgcn_graph.py --event-types purchase`로 그래프 생성 |
| `emb_dim` | 32 | |
| `lr` | 0.002 | |
| `lamda` | 0.02 | |
| `layer` | 2 | uniform 기준 재검증 결과 layer=1도 근접하지만(0.0437), 확정 조합(neg=10/lr=0.002)과는 layer=2가 최선 |
| `optimizer` | Adam (`opt=3`) | |
| `batch` | 10000 | |
| `neg_samples` | 10 | `parse.py --sample_rate` |
| `layer_weight_scheme` | uniform | 표준 LightGCN 방식, `model_LightGCN_tri.py` 기본값 |
| 평가 방식 | 고정 유저 집합(1,465명), K=[5,10,20] | `train_model.py`/`test_model.py`가 그래프 구조 무관 공용 코드라 bipartite에도 자동 적용됨 |
| `split_date` | 2025-08-01 (ALS와 동일) | tri/bipartite/ALS 셋 다 동일 |

tri 쪽 최종 성능: **HR@20≈0.0453**(3회 반복 평균, ±0.0022 — negative sampling이 seed 고정 안 돼 있어 run마다 흔들림, 상세는 리포트 5-4 참고). `configs/LightGCN/params.yaml`에 이 값들이 기본값으로 반영돼 있다 — CLI 인자 없이 `run_lightgcn.py --epoch 300`만 돌려도 이 설정으로 실행된다(단, 그래프 자체는 `--event-types purchase`로 별도 생성 필요, 결과 수치는 반복 실행마다 소폭 흔들릴 수 있음).

## 실험 결과 요약 (완료된 항목)

### 1. 학습 epoch 수 → 300으로 확정

loss가 300 epoch 근처에서 정체하는 걸 확인(2차), 더 늘리는 것보다 다른 하이퍼파라미터 조정이 우선순위가 높다고 판단해 300으로 유지했다.

### 2. u2t event_type 조합 → "구매만"으로 확정

| 조합 | HR@20 |
|---|---|
| **구매만** | **0.0491** |
| 장바구니+구매 | 0.0389 |
| 전부(page_view+장바구니+구매) | 0.0403 |

page_view/장바구니가 오히려 노이즈로 작용 — 구매 신호만 남기는 게 최선이었다.

### 3. 하이퍼파라미터(emb_dim/lr/lamda) → emb_dim=32, lamda=0.02(원래값)로 확정

emb_dim×lr 9개 그리드, lamda 3단계 실험 결과 `emb_dim=32`(원 논문 기본값 128보다 훨씬 작음, ALS의 `factors=16`과 같은 맥락)가 최선이었고, `lamda`는 외부 레퍼런스(RecBole `1e-5`)보다 원래 값(0.02)이 이 데이터엔 더 나았다.

### 4. layer/optimizer/batch/negative sampling → 5차(버그 수정 후) 기준 재확정

`layer=2`, Adam, `batch=10000`은 유지. `neg_samples=10`+`lr=0.002` 조합이 uniform 기본값보다 낫다고 봤으나(1회 관측 +14%), negative sampling이 seed 고정 안 돼 있어 재현성 검증(3회 반복)이 필요했다 — 실제 개선폭은 +5%대(0.0430→0.0453)로 정정. 상세: 리포트 5차 실행 섹션(특히 5-4).

### 5. t2p lift 임계값 / 최소 구매 건수 — 아직 미착수 (우선순위 낮음, 보류)

`LIFT_THRESHOLD = 1.15`, `MIN_PURCHASE_COUNT = 5` — tri 그래프 구조 자체를 바꾸는 거라 영향 범위가 크다. bi vs tri 비교(#34)가 끝난 뒤 필요성을 재검토한다.

## 다음 단계

```text
1. #34: bipartite를 위 확정 설정으로 실행 → tri와 공정 비교
2. (선택) t2p lift 임계값 재검토
```

## 관련 문서

- 전체 실험 과정·수치: `reports/[ML]_LightGCN_tri_전체학습_및_ALS_비교평가_2026-07-08.md`
- ALS 튜닝 사례(참고 패턴): `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb`, `reports/[ML]_#7_ALS_성능개선_및_Threshold_검증_2026-07-03.md`
- t2p lift 설계 배경: `docs/WHY_SEGMENTS.md`, `reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`
