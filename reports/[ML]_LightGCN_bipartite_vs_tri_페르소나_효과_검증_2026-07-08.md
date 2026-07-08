# [ML] LightGCN bipartite vs tri — 페르소나 효과 검증 — 2026-07-08

> **결론**: bipartite 3회 반복까지 완료 — **tri가 bipartite보다 일관되게 좋다** (HR@20 0.0453 vs 0.0398, 범위가 겹치지 않음). 페르소나를 그래프에 결합하면 추천 정확도가 실제로 향상된다는 결론.

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

## 결과

### bipartite 반복 실행 (3회)

| run | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| 1 | 0.0082 | 0.0143 | 0.0416 | 0.0087 |
| 2 | 0.0116 | 0.0218 | 0.0389 | 0.0092 |
| 3 | 0.0075 | 0.0171 | 0.0389 | 0.0089 |
| **평균** | **0.0091** | **0.0177** | **0.0398 (±0.0016)** | **0.0089** |

### 최종 비교 (양쪽 다 3회 평균)

| 모델 | HR@5 | HR@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|
| **bipartite (페르소나 없음)** | 0.0091 | 0.0177 | **0.0398** (범위 0.0389~0.0416) | 0.0089 |
| **tri (페르소나 있음)** | 0.0089 | 0.0232 | **0.0453** (범위 0.0437~0.0478) | 0.0098 |
| ALS (참고, confound 있음) | 0.0171 | 0.0369 | 0.0608 | 0.0152 |

**tri가 bipartite보다 HR@20 기준 +14% 높다(0.0453 vs 0.0398).** 3회씩 반복한 두 그룹의 값 범위가 겹치지 않는다(bipartite 최댓값 0.0416 < tri 최솟값 0.0437) — n=3이라는 한계는 있지만, 이전 negative sampling 재현성 문제(#37 5-4, 범위가 겹쳤던 경우)와 달리 이번엔 방향과 크기가 뚜렷하게 갈린다.

**해석**: 같은 아키텍처, 같은 하이퍼파라미터, 같은 그래프 구조(u2t)에서 페르소나 서브그래프(u2p/t2p) 유무만 다른데 성능 차이가 이만큼 난다는 건, 페르소나가 그래프 임베딩 레벨에서 실제로 유의미한 신호를 준다는 뜻이다. ALS 대비 tri의 격차(1.34배)도 "그래프 기반 방법 자체가 안 맞아서"라는 가설보다 "bipartite보다는 낫지만 ALS만큼은 아니다"로 더 정확하게 설명할 수 있게 됐다.

## 방법론적 제약 및 한계

- **negative sampling에 seed 없음**: `#37` 5-4에서 확인했듯 run마다 HR@20이 흔들린다 — 그래서 tri와 동일하게 bipartite도 3회 반복해서 평균으로 비교했다. bipartite의 표준편차(±0.0016)는 tri(±0.0022)와 비슷한 수준으로, 관측된 격차(+14%)가 이 노이즈보다 뚜렷하게 크다.
- **페르소나(세그먼트)만 통제된 비교**: 그래프 구조와 하이퍼파라미터는 완전히 동일, u2p/t2p 서브그래프 유무만 다르다 — 이 리포트가 유일하게 페르소나의 순수 효과를 보여주는 비교다.
- **n=3의 한계**: 통계적으로 엄밀한 유의성 검정을 하기엔 반복 횟수가 적다. 다만 범위가 전혀 겹치지 않는다는 점에서 방향성은 신뢰할 만하다고 판단했다.
- **ALS 비교는 여전히 confound 있음**: 참고용으로만 병기, 페르소나 효과의 근거로 쓰지 않는다.

## 관련 산출물

- `src/baselines/lgcn3/read_data.py` (수정 — `graph_mode` 파라미터 추가, `_resolve_graph_paths()` 순수 함수 분리, TDD 4건)
- `src/baselines/lgcn3/run_lightgcn.py` (수정 — `--graph-mode bipartite` 지원 추가)
- `tests/test_read_data.py` (신규, 4건)
- `data/outputs/LightGCN/hp_tuning_experiment/bipartite_variance_check/` (3회 반복 결과 백업, gitignore 대상)

## 다음 단계

- [x] ~~bipartite 그래프 로딩 지원 구현~~
- [x] ~~bipartite 3회 반복 실행 및 평가~~
- [ ] `#34` 이슈에 최종 결과 공유
- [ ] 대시보드 Tab2("페르소나 기여도") 근거 데이터로 전달
- [ ] (선택) bipartite가 tri에 졌으므로, Twiddler reranking을 ALS/bipartite 위에 적용해 "재랭킹을 통한 페르소나 효과"도 별도로 검증 — 다음 논의 필요
