# [ML] LightGCN_tri 전체 학습 및 ALS 비교 평가 — 2026-07-08

> **업데이트 (같은 날, 2차 실행)**: 최초 결과(아래 "1차 실행") 이후, 학습 진단용 loss 로깅이 꺼져 있고 평가가 매 epoch 다른 무작위 512명 표본이라 곡선을 신뢰할 수 없다는 문제를 발견해 고쳤다(`train_model.py`/`test_model.py`). 같은 하이퍼파라미터로 재학습한 "2차 실행" 결과를 이 문서 뒷부분에 추가했다 — **측정 문제만 고쳤는데 HR@20이 0.0294 → 0.0403으로 개선**됐다(모델을 바꾼 게 아님). 최신 결론은 [업데이트된 핵심 발견](#핵심-발견-2차-실행-반영) 참고.

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

## 방법론적 제약 및 한계

- **하이퍼파라미터 미튜닝**: 2차 실행도 lr/lamda/layer는 1차와 동일한 원 논문 기본값이다. 이번 결과는 LightGCN_tri의 "잠재 성능"이 아니라 "기본값 성능"이다.
- **단일 실행(seed 1개)**: 학습 초기화·negative sampling에 고정 seed를 안 써서, 1차/2차 실행 간 수치 차이 중 일부는 이 확률성 때문일 수 있다(측정 방식 개선 효과와 완전히 분리되지 않음). ALS도 동일한 한계(`reports/[ML]_#7_ALS_...md`의 "모든 실험이 동일한 test set을 반복 조회")를 가지고 있다.
- **validation split 없음**: ALS와 동일하게 test set을 그대로 봤다 — 배포 전 재검증 필요(ALS 리포트의 한계와 동일).
- **페르소나(세그먼트) 효과는 이 리포트로 판단할 수 없다**: 이 결과는 LightGCN(tri, 페르소나 포함) vs ALS(페르소나 없음) 비교라 "모델 종류"와 "페르소나 유무"가 섞여 있다 — 정확히 #34에서 지적한 confound 문제. 페르소나 자체의 효과를 보려면 #34(LightGCN bipartite vs tri)가 필요하다. 이번에 고친 평가 방식(고정 유저셋, loss 로깅)은 `train_model.py`/`test_model.py`(그래프 구조 무관 공용 코드)에 있어 bipartite 실행에도 자동 적용된다 — #34 코멘트에 공유해둠.

## 관련 산출물

- `src/baselines/lgcn3/evaluate_lightgcn.py` (신규)
- `tests/test_evaluate_lightgcn.py` (신규, 12건)
- `configs/LightGCN/params.yaml` (`eval_k_list` 추가)
- `src/baselines/lgcn3/train_model.py`, `test_model.py` (수정 — loss 로깅 추가, 평가 유저 고정)
- `docs/LIGHTGCN_TRI_TUNING_PLAN.md` (신규 — 다음 개선 후보 우선순위와 bipartite 비교 조건 정리)
- `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv` (2,000,000행, gitignore 대상 로컬 산출물, 2차 실행 결과로 덮어씀)
- `data/outputs/LightGCN/eval_results.csv` (gitignore 대상 로컬 산출물)
- `logs/LightGCN/run_lightgcn_train_log.xlsx` (`Loss` 시트 신규 — gitignore 대상 로컬 산출물)

## 권장 다음 단계

우선순위와 상세 근거는 `docs/LIGHTGCN_TRI_TUNING_PLAN.md` 참고. 요약:

- [x] ~~학습 진단(loss 로깅, 평가 노이즈)~~ — 이번 리포트에서 완료
- [ ] u2t event_type 조합 실험 (전부 포함 vs add_to_cart+구매 vs 구매만) → HR@20 비교, #29에서 이미 파라미터화해둔 부분 — loss가 정체됐으니 epoch 수 늘리기보다 우선순위 높음
- [ ] 하이퍼파라미터 튜닝 (ALS의 그리드서치 노트북 패턴 참고) — lr/lamda/layer/emb_dim
- [ ] #34: LightGCN bipartite(#35) vs tri 비교로 페르소나 순수 효과 분리 측정 — 위 두 항목에서 정해지는 조합(event_type, 하이퍼파라미터)을 bipartite에도 동일 적용해야 공정한 비교
- [ ] rec-system 레포 연동 시 `PRED_MAIN_RECOMMEND.csv`에 `twiddler`/`user_type` 컬럼 추가 필요 (rec-system #4 스키마 요구사항, 현재 미포함)
