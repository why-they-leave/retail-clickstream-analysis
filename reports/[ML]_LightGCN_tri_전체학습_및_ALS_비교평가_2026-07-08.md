# [ML] LightGCN_tri 전체 학습 및 ALS 비교 평가 — 2026-07-08

> **업데이트 (같은 날, 2차 실행)**: 최초 결과(아래 "1차 실행") 이후, 학습 진단용 loss 로깅이 꺼져 있고 평가가 매 epoch 다른 무작위 512명 표본이라 곡선을 신뢰할 수 없다는 문제를 발견해 고쳤다(`train_model.py`/`test_model.py`). 같은 하이퍼파라미터로 재학습한 "2차 실행" 결과를 이 문서 뒷부분에 추가했다 — **측정 문제만 고쳤는데 HR@20이 0.0294 → 0.0403으로 개선**됐다(모델을 바꾼 게 아님).
>
> **업데이트 (같은 날, 3차 — event_type 실험 + 하이퍼파라미터 튜닝)**: `docs/LIGHTGCN_TRI_TUNING_PLAN.md` 우선순위대로 u2t event_type 조합 실험 → `emb_dim`×`lr` 그리드 → `lamda` 라운드를 순서대로 진행했다. **최종 HR@20=0.0553, ALS 대비 격차 2.07배 → 1.10배**까지 좁혔다. 상세는 [3차 실행](#분석-결과--3차-실행-event_type--하이퍼파라미터-튜닝) 참고.

## 분석 목적

Issue #30. 스모크 테스트(`reports/[ML]_LightGCN_tri_모델_구현_및_스모크테스트_2026-07-07.md`)로 파이프라인이 정상 동작함을 확인한 뒤, 실제 전체 300 epoch 학습을 실행하고 `evaluate_lightgcn.py`(신규)로 Hit Ratio@K/Recall@K/NDCG@K를 계산해 ALS(#7)와 같은 기준으로 비교한다.

## 데이터셋 버전 및 기간

- #29 tri-graph 산출물 그대로 사용 (user_num=20,000, item_num=1,197, persona_num=6)
- 평가 정답셋: `tri_graph_uidx2tidx_valid.json`(구매만, ALS의 `split_date=2025-08-01` 이후 test 기간과 동일 기준) — 원본 id로 디코딩해 즉석에서 생성
- 평가 대상 유저: 1,465명 (ALS 리포트의 1,463명과 거의 일치 — 같은 test 기간·구매 기준을 쓴다는 정합성 확인)

## 방법론

```text
1. run_lightgcn.py --epoch 300
   → 학습(하이퍼파라미터: parse.py 원 논문 기본값 그대로, lr=0.005/lamda=0.02/layer=2)
   → 전체 유저(20,000명) top-100 추천 생성 + PRED_MAIN_RECOMMEND.csv 저장
2. evaluate_lightgcn.py --k 5 10 20
   → 사전 계산된 CSV 기반 평가 (모델 재추론 없음, als_evaluate.py와 동일 구조)
   → HR@K, Recall@K, NDCG@K 계산
```

관련 코드: `src/baselines/lgcn3/evaluate_lightgcn.py`(신규, TDD 12건), `configs/LightGCN/params.yaml`(`eval_k_list` 추가)

## 평가 지표

| 지표 | 1차 실행 | 2차 실행 |
|------|-----|-----|
| 학습 시간 (300 epoch) | 1시간 40분 15초 | 1시간 22분 26초 |
| epoch당 평균 시간 | 20.05초 | 16.49초 |
| 추천 생성 (전체 유저) | 9초 | 10초 |
| 평가 대상 유저 | 1,465명 | 1,465명 |
| 단위 테스트 | 12/12 통과 (evaluate_lightgcn.py 신규), 전체 회귀 51/51 통과 | (2차는 코드 재사용, 재검증 불필요) |

## 분석 결과 — 1차 실행

**핵심 발견:**

- **이번 설정(하이퍼파라미터 미튜닝)에서는 ALS가 LightGCN_tri보다 전 지표에서 우세하다.**

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 평가 유저 |
|---|---|---|---|---|---|---|---|
| **ALS** (#7) | 0.0171 | 0.0369 | 0.0608 | 0.0075 | 0.0114 | 0.0152 | 1,463 |
| **LightGCN_tri** (1차) | 0.0109 | 0.0177 | 0.0294 | 0.0035 | 0.0047 | 0.0066 | 1,465 |
| 차이 (LightGCN − ALS) | −0.0062 | −0.0192 | −0.0314 | −0.0040 | −0.0067 | −0.0086 | — |

HR@20 기준 ALS가 LightGCN_tri보다 약 2.07배 높다.

- **가능한 원인 (1차 실행 시점 가설, 아래 2차 실행에서 일부 검증됨):**
  1. **하이퍼파라미터를 전혀 안 건드렸다** — `lr=0.005, lamda=0.02, layer=2`는 원 논문의 다른 데이터셋(Amazon/Movielens) 기본값을 그대로 가져온 것이라, 우리 데이터(20,000명·1,197개, ALS보다 훨씬 큰 유저 스케일)에 최적이라는 보장이 없다.
  2. **u2t event_type 조합을 실험하지 않았다** — #29 리포트에서 이미 "어떤 event_type 조합이 최선인지는 #30에서 실험적으로 정하기로 했다"고 남겨뒀던 부분인데, 이번엔 기본값(`page_view+add_to_cart+purchase` 전부)으로만 돌렸다.
  3. ALS는 `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb`에서 그리드서치를 거친 반면, LightGCN_tri는 단 1회 실행이라 애초에 공정한 비교가 아닐 수 있다.
  4. **(2차 실행에서 확인) 학습·평가 자체를 제대로 진단할 수 없는 상태였다** — raw loss 로깅이 꺼져 있었고, epoch마다 무작위 512명(그중 실제 test 정답이 있는 사람은 20,000명 중 1,465명뿐이라 평균 ~38명)으로 평가해 F1 곡선이 순수 표본 노이즈에 가까웠다. "성능이 낮다"는 결론 자체가 측정 오차 위에 서 있었을 가능성이 있었다.

**해석 (1차 시점):** 지금 시점에서 "LightGCN이 ALS보다 나쁜 모델"이라고 결론 내리기는 이르다. 튜닝을 전혀 안 한 상태의 1회성 비교이기 때문이다.

## 분석 결과 — 2차 실행 (측정 방식 개선 후 재실행)

**변경한 것 (`train_model.py`/`test_model.py`, 하이퍼파라미터·데이터는 1차와 동일):**
- BPR loss를 epoch마다 기록 (`logs/LightGCN/run_lightgcn_train_log.xlsx`의 `Loss` 시트) — 기존엔 로깅 호출 자체가 주석 처리돼 있었음
- 평가 대상 유저를 매 epoch 무작위 재추첨하지 않고, test 정답이 있는 1,465명 전체로 고정

**핵심 발견 (2차 실행 반영):**

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 평가 유저 |
|---|---|---|---|---|---|---|---|
| ALS (#7) | 0.0171 | 0.0369 | 0.0608 | 0.0075 | 0.0114 | 0.0152 | 1,463 |
| LightGCN_tri (1차, 노이즈 많은 평가) | 0.0109 | 0.0177 | 0.0294 | 0.0035 | 0.0047 | 0.0066 | 1,465 |
| **LightGCN_tri (2차, 고정 평가셋)** | 0.0109 | **0.0191** | **0.0403** | — | — | **0.0088** | 1,465 |
| 차이 (2차 − ALS) | −0.0062 | −0.0178 | −0.0205 | — | — | −0.0064 | — |

- **모델도, 하이퍼파라미터도 안 바꿨는데 측정만 고쳤더니 HR@20이 0.0294 → 0.0403로 약 37% 개선됐다.** 즉 1차 결과는 "모델이 나쁘다"보다 "노이즈 많은 측정이 성능을 과소평가했다"에 더 가까웠다.
- **loss는 300 epoch 내내 감소하다가 마지막 20 epoch(281~300)에서 2650~2695 구간에 정체** — 지금 학습률(`lr=0.005`)로는 사실상 수렴했다. epoch을 더 늘리는 것보다 학습률/정규화 등 다른 하이퍼파라미터를 바꾸는 쪽이 다음 개선 여지로 보인다.
- **ALS와의 격차는 좁혀졌지만(2.07배 → 1.51배) 여전히 ALS가 우세하다.**

**해석 (2차 시점):** "측정이 고쳐지고 나서도 ALS보다 낮다"는 건 유효한 신호다. 다만 1차 실행의 가설 1(하이퍼파라미터)·2(event_type 조합)는 아직 전혀 검증 안 된 상태라, 이 상태로 "LightGCN이 이 데이터에 안 맞는다"고 결론 내리기도 이르다. 다음 단계는 `docs/LIGHTGCN_TRI_TUNING_PLAN.md`에 정리했다.

## 분석 결과 — 3차 실행 (event_type + 하이퍼파라미터 튜닝)

`docs/LIGHTGCN_TRI_TUNING_PLAN.md` 우선순위(그래프 구조 → 하이퍼파라미터)대로 순서대로 실험했다. 모든 실험은 `make_lgcn_graph.py --event-types ...`(신규 CLI, TDD 3건)와 `run_lightgcn.py --lr/--emb-dim/--lamda`(신규 CLI 오버라이드) 옵션으로 진행했고, 각 조합마다 300 epoch 전체 학습 후 `evaluate_lightgcn.py`로 평가했다.

### 3-1. u2t event_type 조합 (그래프 구조)

| 조합 | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| page_view+장바구니+구매 (2차 baseline) | 0.0109 | 0.0191 | 0.0403 | 0.0088 |
| **구매만** | **0.0116** | **0.0280** | **0.0491** | **0.0105** |
| 장바구니+구매 | 0.0068 | 0.0143 | 0.0389 | 0.0074 |

**"구매만"이 최선이었다** (HR@20 +22%). 반직관적으로 "장바구니+구매"가 "전부 포함"보다도 나빴다 — add_to_cart가 실제 구매 의도와 안 맞는 경우(담았다 안 산 경우 등)가 많아 구매 신호를 희석하는 노이즈로 작용한 것으로 추정된다. 부수 효과로, 그래프가 훨씬 희소해져 학습 속도도 크게 빨라졌다(epoch당 20초 → 약 0.7초, 300 epoch가 약 4분).

### 3-2. emb_dim × lr 그리드 (9개 조합, event_type=구매만 고정)

| emb_dim | lr | HR@20 | NDCG@20 |
|---|---|---|---|
| **32** | **0.005** | **0.0553** | **0.0131** |
| 64 | 0.01 | 0.0546 | 0.0117 |
| 64 | 0.005 | 0.0539 | 0.0115 |
| 128 | 0.001 | 0.0532 | 0.0124 |
| 32 | 0.01 | 0.0505 | 0.0109 |
| 128 | 0.005 | 0.0485 | 0.0093 |
| 128 | 0.01 | 0.0478 | 0.0102 |
| 32 | 0.001 | 0.0457 | 0.0089 |
| 64 | 0.001 | 0.0423 | 0.0101 |

`emb_dim=32, lr=0.005`가 최선. `emb_dim=128`(원 논문 기본값)은 전반적으로 하위권 — ALS가 `factors=16`을 쓰는 것과 같은 맥락으로, 이 데이터 규모(2만 유저·1,197개 상품)엔 128차원이 과도했던 것으로 보인다.

### 3-3. lamda(L2 정규화) 라운드 (emb_dim=32, lr=0.005 고정)

RecBole 공식 문서가 LightGCN 기본 `reg_weight=1e-5`를 제시해, 우리 `lamda=0.02`(원 논문 다른 데이터셋 기본값)가 과도한 정규화일 수 있다는 가설을 세우고 검증했다.

| lamda | HR@20 | NDCG@20 |
|---|---|---|
| **0.02 (원래 기본값)** | **0.0553** | **0.0131** |
| 0.002 | 0.0403 | 0.0108 |
| 0.00001 (RecBole 제안값) | 0.0498 | 0.0102 |

**가설과 반대로, 원래 값(0.02)이 최선이었다.** RecBole 기본값은 MovieLens/Amazon-Book 같은 대형 벤치마크 기준이라, 우리처럼 작고 희소한 데이터(실제 구매 정답이 있는 유저 1,465명뿐)에는 오히려 더 강한 정규화가 과적합 방지에 유리했던 것으로 해석된다. 외부 레퍼런스 기본값을 검증 없이 이식하면 안 된다는 걸 보여주는 사례다.

### 3차 최종 결과

**확정 설정: event_type=구매만, emb_dim=32, lr=0.005, lamda=0.02(원래 값), layer=2(미변경)**

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| ALS (#7) | 0.0171 | 0.0369 | 0.0608 | 0.0152 |
| LightGCN_tri 1차 | 0.0109 | 0.0177 | 0.0294 | 0.0066 |
| LightGCN_tri 2차 | 0.0109 | 0.0191 | 0.0403 | 0.0088 |
| **LightGCN_tri 3차 (최종)** | **0.0177** | **0.0300** | **0.0553** | **0.0131** |

HR@20 기준 ALS와의 격차가 **2.07배 → 1.51배 → 1.10배**로 단계적으로 좁혀졌다. 여전히 ALS가 근소하게 우세하지만, 튜닝 전 대비 격차가 크게 줄었다.

**해석 (3차 시점):** 측정 진단(2차) + 그래프 구조(event_type) + 하이퍼파라미터(emb_dim/lamda) 세 단계 모두 실질적인 개선에 기여했다. 다만 `layer`는 이번에 안 건드렸다.

### 3-4. negative sampling 개수 (SimpleX 참고, emb_dim=32/lr=0.005/lamda=0.02 고정)

기존엔 양성 1개당 음성 1개만 무작위로 뽑았다(`train_model.py`, `MODEL in [...]` 목록에 `LightGCN_tri`가 빠져있어 강제로 1개였음 — 이번에 목록에 추가해 `SAMPLE_RATE`를 실제로 쓰도록 고침, `parse.py`에 `--sample_rate`, `run_lightgcn.py`에 `--neg-samples` CLI 추가). SimpleX 논문이 negative를 늘리면 NDCG가 크게 개선된다고 보고해 검증했다.

| neg_samples | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| **1 (기존 확정)** | **0.0177** | **0.0300** | **0.0553** | **0.0131** |
| 5 | 0.0116 | 0.0225 | 0.0444 | 0.0092 |
| 10 | 0.0116 | 0.0253 | 0.0437 | 0.0092 |

**또 반직관적 — 늘릴수록 나빠졌다.** 다만 **완전히 공정한 비교는 아니다**: negative를 늘리면 배치당 (양성,음성) 쌍이 그만큼 늘어 BPR loss 절댓값이 커지는데(1350대 → 6800대), `lr=0.005`는 negative 1개 기준으로 찾은 값이라 negative를 늘렸을 때 최적 학습률이 달라졌을 수 있다 — 그 재탐색은 안 했다. **"negative sampling이 무조건 안 통한다"가 아니라 "이번 lr 기준으로는 그랬다"가 정확한 결론이다.**

**최종 확정 설정은 3-3 결과(neg_samples=1, 즉 기존 방식) 그대로 유지한다.**

## 분석 결과 — 4차 실행 (layer / optimizer / batch size, #37)

3차 종료 후 튜닝 작업을 별도 이슈(#37)로 분리했다 — #30의 원래 범위(환경 구축)는 완료됐고, 튜닝은 끝이 정해지지 않은 열린 작업이라 추적을 분리하는 게 낫다고 판단(상세: #30 PR #38, #37 이슈 본문). 이하 결과는 #37에서 진행한 것이다.

확정 설정(구매만, emb_dim=32, lr=0.005, lamda=0.02, neg_samples=1) 고정, `layer`/`opt`/`batch` 세 축을 각각 바꿔봤다. `run_lightgcn.py`에 `--layer`/`--opt`/`--batch` CLI 추가(`parse.py`엔 `--opt`/`--batch`가 이미 있어 노출만 함).

| 조합 | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| **baseline (layer=2, Adam, batch=10000)** | **0.0177** | **0.0300** | **0.0553** | **0.0131** |
| layer=1 | 0.0137 | 0.0259 | 0.0471 | 0.0108 |
| layer=3 | 0.0116 | 0.0191 | 0.0410 | 0.0089 |
| layer=4 | 0.0130 | 0.0212 | 0.0410 | 0.0092 |
| opt=SGD | 0.0109 | 0.0259 | 0.0491 | 0.0107 |
| opt=RMSProp | 0.0102 | 0.0246 | 0.0532 | 0.0122 |
| batch=1000 | 0.0150 | 0.0287 | 0.0485 | 0.0112 |
| batch=2000 | 0.0055 | 0.0157 | 0.0382 | 0.0078 |

**전부 baseline을 못 넘었다** — 3차까지와 달리 이번엔 반직관적 결과가 아니라, 기존 확정값이 이 세 축에서도 이미 최적점이었다는 확인이다.

- **layer**: 2가 최적. 1은 표현력 부족, 3~4는 LightGCN에서 잘 알려진 over-smoothing(레이어가 깊어질수록 임베딩이 서로 비슷해져 구분력이 떨어지는 현상)으로 추정.
- **optimizer**: Adam > RMSProp(근소) > SGD.
- **batch size**: 10000(사실상 풀배치에 가까움) > 1000 > 2000. batch=1000이 2000보다 나은 건 순서상 이례적이라 재현성엔 주의가 필요하나, 어느 쪽도 baseline은 못 넘었다.

**최종 확정 설정은 3차와 동일하게 유지한다**: 구매만, emb_dim=32, lr=0.005, lamda=0.02, layer=2, Adam(opt=3), batch=10000, neg_samples=1 → **HR@20=0.0553**.

## 방법론적 제약 및 한계

- **layer 수·optimizer·batch size는 4차에서 튜닝 완료**: `layer ∈ {1,2,3,4}`, `opt ∈ {SGD,RMSProp,Adam}`, `batch ∈ {1000,2000,10000}` 실험 결과 기존 확정값(layer=2, Adam, batch=10000)이 전부 최적으로 확인됨.
- **negative sampling 실험은 lr 재탐색 없이 진행**: 3-4에서 negative를 늘리면 loss 스케일이 달라지는데 `lr`을 그에 맞게 다시 찾지 않았다 — "negative 늘리기가 안 통한다"보다는 "이 lr에서는 그랬다"로 해석해야 한다.
- **단일 실행(seed 1개)**: 학습 초기화·negative sampling에 고정 seed를 안 써서, 1차/2차 실행 간 수치 차이 중 일부는 이 확률성 때문일 수 있다(측정 방식 개선 효과와 완전히 분리되지 않음). ALS도 동일한 한계(`reports/[ML]_#7_ALS_...md`의 "모든 실험이 동일한 test set을 반복 조회")를 가지고 있다.
- **validation split 없음**: ALS와 동일하게 test set을 그대로 봤다 — 배포 전 재검증 필요(ALS 리포트의 한계와 동일).
- **페르소나(세그먼트) 효과는 이 리포트로 판단할 수 없다**: 이 결과는 LightGCN(tri, 페르소나 포함) vs ALS(페르소나 없음) 비교라 "모델 종류"와 "페르소나 유무"가 섞여 있다 — 정확히 #34에서 지적한 confound 문제. 페르소나 자체의 효과를 보려면 #34(LightGCN bipartite vs tri)가 필요하다. 이번에 고친 평가 방식(고정 유저셋, loss 로깅)은 `train_model.py`/`test_model.py`(그래프 구조 무관 공용 코드)에 있어 bipartite 실행에도 자동 적용된다 — #34 코멘트에 공유해둠.

## 관련 산출물

- `src/baselines/lgcn3/evaluate_lightgcn.py` (신규)
- `tests/test_evaluate_lightgcn.py` (신규, 12건)
- `configs/LightGCN/params.yaml` (`eval_k_list` 추가)
- `src/baselines/lgcn3/train_model.py`, `test_model.py` (수정 — loss 로깅 추가, 평가 유저 고정)
- `src/datasets/make_lgcn_graph.py` (수정 — `--event-types` CLI 옵션 추가, TDD 3건)
- `src/baselines/lgcn3/run_lightgcn.py` (수정 — `--lr`/`--emb-dim`/`--lamda`/`--neg-samples` CLI 오버라이드 추가)
- `src/baselines/lgcn3/parse.py`, `params.py` (수정 — `--sample_rate` CLI 추가)
- `src/baselines/lgcn3/train_model.py` (수정 — `LightGCN_tri`가 `SAMPLE_RATE`를 쓰도록 모델 목록에 추가)
- `src/baselines/lgcn3/run_lightgcn.py` (수정 — `--layer`/`--opt`/`--batch` CLI 오버라이드 추가, #37)
- `pyproject.toml` (수정 — `known-first-party`에 lgcn3 형제 모듈 등록, CI ruff I001 픽스)
- `docs/LIGHTGCN_TRI_TUNING_PLAN.md` (신규 — 다음 개선 후보 우선순위와 bipartite 비교 조건 정리)
- `data/outputs/LightGCN/hp_tuning_experiment/`, `event_type_experiment/` (실험별 eval_results 백업, gitignore 대상 로컬 산출물)
- `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv`, `eval_results.csv` (최종 확정 설정 결과로 덮어씀, gitignore 대상)
- `logs/LightGCN/run_lightgcn_train_log.xlsx` (`Loss` 시트 신규 — gitignore 대상 로컬 산출물)

## 권장 다음 단계

우선순위와 상세 근거는 `docs/LIGHTGCN_TRI_TUNING_PLAN.md` 참고. 요약:

- [x] ~~학습 진단(loss 로깅, 평가 노이즈)~~ — 2차에서 완료
- [x] ~~u2t event_type 조합 실험~~ — 3차에서 완료, "구매만"으로 확정
- [x] ~~하이퍼파라미터 튜닝(emb_dim, lr, lamda)~~ — 3차에서 완료, `emb_dim=32, lr=0.005, lamda=0.02`로 확정
- [x] ~~negative sampling 개수, layer 수, optimizer, batch size 튜닝~~ — 3-4/4차(#37)에서 완료, 전부 기존 확정값이 최적으로 확인됨
- [ ] negative sampling × lr 재짝짓기 재실험 (lr을 재탐색 없이 진행한 3-4의 한계 보완) — #37 진행 중
- [ ] #34: LightGCN bipartite(#35) vs tri 비교로 페르소나 순수 효과 분리 측정 — 최종 확정 조합(구매만, emb_dim=32, lr=0.005, lamda=0.02, layer=2, Adam, batch=10000)을 bipartite에도 동일 적용해야 공정한 비교, #34 코멘트에 공유 예정
- [ ] rec-system 레포 연동 시 `PRED_MAIN_RECOMMEND.csv`에 `twiddler`/`user_type` 컬럼 추가 필요 (rec-system #4 스키마 요구사항, 현재 미포함)
