# [ML] LightGCN_tri 전체 학습 및 ALS 비교 평가 — 2026-07-08

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

| 지표 | 값 |
|------|-----|
| 학습 시간 (300 epoch) | 1시간 40분 15초 |
| epoch당 평균 시간 | 20.05초 |
| 추천 생성 (전체 유저) | 9초 |
| 평가 대상 유저 | 1,465명 |
| 단위 테스트 | 12/12 통과 (evaluate_lightgcn.py 신규), 전체 회귀 51/51 통과 |

## 분석 결과

**핵심 발견:**

- **이번 설정(하이퍼파라미터 미튜닝)에서는 ALS가 LightGCN_tri보다 전 지표에서 우세하다.**

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | 평가 유저 |
|---|---|---|---|---|---|---|---|
| **ALS** (#7) | 0.0171 | 0.0369 | 0.0608 | 0.0075 | 0.0114 | 0.0152 | 1,463 |
| **LightGCN_tri** (이번) | 0.0109 | 0.0177 | 0.0294 | 0.0035 | 0.0047 | 0.0066 | 1,465 |
| 차이 (LightGCN − ALS) | −0.0062 | −0.0192 | −0.0314 | −0.0040 | −0.0067 | −0.0086 | — |

HR@20 기준 ALS가 LightGCN_tri보다 약 2.07배 높다.

- **가능한 원인 (확정 아님, 다음 단계에서 검증 필요):**
  1. **하이퍼파라미터를 전혀 안 건드렸다** — `lr=0.005, lamda=0.02, layer=2`는 원 논문의 다른 데이터셋(Amazon/Movielens) 기본값을 그대로 가져온 것이라, 우리 데이터(20,000명·1,197개, ALS보다 훨씬 큰 유저 스케일)에 최적이라는 보장이 없다.
  2. **u2t event_type 조합을 실험하지 않았다** — #29 리포트에서 이미 "어떤 event_type 조합이 최선인지는 #30에서 실험적으로 정하기로 했다"고 남겨뒀던 부분인데, 이번엔 기본값(`page_view+add_to_cart+purchase` 전부)으로만 돌렸다.
  3. ALS는 `notebooks/20260703_ML_als_hyperparameter_tuning.ipynb`에서 그리드서치를 거친 반면, LightGCN_tri는 단 1회 실행이라 애초에 공정한 비교가 아닐 수 있다.

**해석:** 지금 시점에서 "LightGCN이 ALS보다 나쁜 모델"이라고 결론 내리기는 이르다. 튜닝을 전혀 안 한 상태의 1회성 비교이기 때문이다. 다만 이 결과 자체는 유효한 baseline 기록으로 남기고, 다음 단계에서 하이퍼파라미터/이벤트 조합을 조정하며 재평가하는 것이 맞다.

## 방법론적 제약 및 한계

- **하이퍼파라미터 미튜닝**: 이번 결과는 LightGCN_tri의 "잠재 성능"이 아니라 "기본값 성능"이다.
- **단일 실행(seed 1개)**: ALS도 동일한 한계(`reports/[ML]_#7_ALS_...md`의 "모든 실험이 동일한 test set을 반복 조회")를 가지고 있어 비교 자체의 공정성은 유지되지만, 두 모델 다 재현성 변동폭을 모른다.
- **validation split 없음**: ALS와 동일하게 test set을 그대로 봤다 — 배포 전 재검증 필요(ALS 리포트의 한계와 동일).
- **페르소나(세그먼트) 효과는 이 리포트로 판단할 수 없다**: 이 결과는 LightGCN(tri, 페르소나 포함) vs ALS(페르소나 없음) 비교라 "모델 종류"와 "페르소나 유무"가 섞여 있다 — 정확히 #34에서 지적한 confound 문제. 페르소나 자체의 효과를 보려면 #34(LightGCN bipartite vs tri)가 필요하다.

## 관련 산출물

- `src/baselines/lgcn3/evaluate_lightgcn.py` (신규)
- `tests/test_evaluate_lightgcn.py` (신규, 12건)
- `configs/LightGCN/params.yaml` (`eval_k_list` 추가)
- `data/outputs/LightGCN/PRED_MAIN_RECOMMEND.csv` (2,000,000행, gitignore 대상 로컬 산출물)
- `data/outputs/LightGCN/eval_results.csv` (gitignore 대상 로컬 산출물)

## 권장 다음 단계

- [ ] 하이퍼파라미터 튜닝 (ALS의 그리드서치 노트북 패턴 참고) — lr/lamda/layer/emb_dim
- [ ] u2t event_type 조합 실험 (전부 포함 vs add_to_cart+구매 vs 구매만) → HR@20 비교, #29에서 이미 파라미터화해둔 부분
- [ ] #34: 시간 여유가 되면 LightGCN bipartite(#35) vs tri 비교로 페르소나 순수 효과 분리 측정
- [ ] rec-system 레포 연동 시 `PRED_MAIN_RECOMMEND.csv`에 `twiddler`/`user_type` 컬럼 추가 필요 (rec-system #4 스키마 요구사항, 현재 미포함)
