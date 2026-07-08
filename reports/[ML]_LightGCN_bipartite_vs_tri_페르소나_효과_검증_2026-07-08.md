# [ML] LightGCN bipartite vs tri — 페르소나 효과 검증 — 2026-07-08

> 진행 중인 리포트입니다. 실험이 끝날 때마다 갱신합니다.

## 분석 목적

Issue #34. `#37`에서 tri(페르소나 결합) 쪽 튜닝이 끝났다. 같은 하이퍼파라미터로 bipartite(페르소나 미결합, #35에서 그래프 데이터 준비 완료)를 학습·평가해서, **페르소나를 그래프에 결합하는 게 실제로 추천 정확도에 도움이 되는지**를 검증한다.

ALS vs LightGCN 비교(기존 리포트)는 알고리즘 계열과 페르소나 유무가 동시에 바뀌는 confounded comparison이라 페르소나 효과의 근거가 될 수 없다 — bipartite vs tri만이 페르소나 하나만 통제된 비교다.

## 데이터셋 버전 및 기간

- u2t(구매 이력) 그래프는 tri와 완전히 동일한 파일(`tri_graph_uidx2tidx_train.json`, 구매만, #37에서 확정) 공유
- bipartite는 u2p/t2p가 전부 빈 매핑(`bipartite_graph_uidx2pidx.json`, `bipartite_graph_tidx2pidx.json`, #35에서 생성) — `persona_num=0`
- 평가 정답셋: tri와 동일 (`tri_graph_uidx2tidx_valid.json`), 평가 유저 1,465명
- split_date: 2025-08-01 (ALS/tri와 동일)

## 방법론

`#37` 최종 확정 하이퍼파라미터를 그대로 적용 — `configs/LightGCN/params.yaml` 기본값 그대로 사용.

```
event_type=구매만, emb_dim=32, lr=0.002, lamda=0.02, layer=2,
optimizer=Adam, batch=10000, neg_samples=10, layer_weight_scheme=uniform
```

```text
1. read_data.py에 graph_mode 파라미터 추가 (tri/bipartite 분기, TDD)
2. run_lightgcn.py --graph-mode bipartite 지원 추가
3. 스모크 테스트(2 epoch) 통과 확인
4. 전체 300 epoch 학습 3회 반복 (negative sampling에 seed가 없어 run-to-run 변동이 있다는 걸
   #37 5-4에서 확인했으므로, tri와 동일하게 bipartite도 3회 평균으로 비교)
5. evaluate_lightgcn.py로 HR@K/Recall@K/NDCG@K 계산
```

## 결과 (진행 중)

### bipartite 반복 실행

| run | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| 1 | 0.0082 | 0.0143 | 0.0416 | 0.0087 |
| 2 | (진행 중) | | | |
| 3 | (예정) | | | |

### 잠정 비교 (run 1 기준, 아직 확정 아님)

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| bipartite (페르소나 없음), run1 | 0.0082 | 0.0143 | 0.0416 | 0.0087 |
| tri (페르소나 있음), 3회 평균 | 0.0089 | 0.0232 | **0.0453** (±0.0022) | 0.0098 |
| ALS (참고, confound 있음) | 0.0171 | 0.0369 | 0.0608 | 0.0152 |

tri가 bipartite run1보다 HR@20 기준 +9% 높다(0.0453 vs 0.0416) — 방향상 "페르소나가 도움된다"는 결론이지만, bipartite는 아직 1회 관측이라 확정 전이다.

## 방법론적 제약 및 한계 (진행 중, 계속 갱신 예정)

- **negative sampling에 seed 없음**: `#37` 5-4에서 확인했듯 run마다 HR@20이 ±0.002~0.0025 정도 흔들린다 — bipartite도 3회 반복해서 평균으로 비교해야 신뢰할 수 있다 (진행 중).
- **페르소나(세그먼트)만 통제된 비교**: 그래프 구조와 하이퍼파라미터는 완전히 동일, u2p/t2p 서브그래프 유무만 다르다.

## 다음 단계

- [ ] bipartite 2, 3회차 실행 및 평가
- [ ] 3회 평균 기준 최종 비교표 확정
- [ ] `#34` 이슈에 결과 공유
- [ ] tri가 이기면(페르소나 효과 확인) → 결론 정리, bipartite가 이기면(또는 유의미한 차이 없으면) → Twiddler reranking을 bipartite 위에 적용하는 실험으로 전환
