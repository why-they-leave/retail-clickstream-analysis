# [ML] LightGCN bipartite vs tri — 페르소나 효과 검증 — 2026-07-08

> **최종 결론**: emb_dim=32(tri-tuned 설정)에서는 tri가 bipartite보다 뚜렷이 좋았다(HR@20 0.0453 vs 0.0398, +14%). 그런데 둘 다 emb_dim=64로 다시 재보니 차이가 사실상 사라졌다(tri 0.0505 vs bipartite 0.0496, 둘 다 표준편차 안에서 겹침). **즉 "페르소나가 도움된다"는 결론은 하이퍼파라미터에 크게 좌우되는, 약하고 불안정한 효과다** — 뚜렷한 효과라고 주장하기 어렵다. 상세 근거는 [5. 로버스트니스 체크](#5-로버스트니스-체크--tri-tuned-설정만으로-평가해도-되는가) 참고.

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

**해석 (emb_dim=32 시점)**: 같은 아키텍처, 같은 하이퍼파라미터, 같은 그래프 구조(u2t)에서 페르소나 서브그래프(u2p/t2p) 유무만 다른데 성능 차이가 이만큼 난다는 건, 페르소나가 그래프 임베딩 레벨에서 유의미한 신호를 준다는 뜻으로 보였다. **다만 이 해석은 아래 5번 로버스트니스 체크에서 뒤집혔다 — emb_dim=32는 우연히 tri에만 유리한 지점이었다.**

## 5. 로버스트니스 체크 — tri-tuned 설정만으로 평가해도 되는가

**우려**: 위 비교는 tri에 맞춰 튜닝된 하이퍼파라미터(emb_dim=32 등)를 bipartite에도 그대로 적용한 것이다. bipartite는 페르소나 노드가 없어 그래프가 더 단순한데, 더 큰 임베딩 차원이 자체적으로는 더 잘 맞을 수 있다 — 그렇다면 위 "+14%"는 페르소나 효과가 아니라 "bipartite에게 불리한 설정을 썼기 때문"일 수 있다.

### 5-1. bipartite에 다른 하이퍼파라미터 시도 (1회씩)

| 조합 | HR@20 |
|---|---|
| emb16, layer2 | 0.0416 |
| emb32, layer1 | 0.0280 |
| **emb64, layer2** | **0.0485** |

`emb_dim=64`가 tri-tuned 기준값(0.0398)보다 훨씬 좋고, tri 평균(0.0453)까지 넘어섰다 — 우려가 맞았다는 신호. 3회 반복으로 확인했다.

### 5-2. bipartite emb_dim=64, 3회 반복

| run | HR@20 |
|---|---|
| 1 | 0.0485 |
| 2 | 0.0478 |
| 3 | 0.0526 |
| **평균** | **0.0496 (±0.0026)** |

tri 평균(0.0453)보다 확실히 높다 — bipartite가 tri를 이겼다.

### 5-3. 그런데 tri도 emb_dim=64로 다시 확인해보니

emb_dim=64 비교도 그 자체로 confound다(bipartite만 emb_dim을 바꿨으니까) — tri도 같은 조건(emb_dim=64)에서 재확인해야 진짜 공정한 비교다.

| run | HR@20 |
|---|---|
| 1 | 0.0451 |
| 2 | 0.0553 |
| 3 | 0.0512 |
| **평균** | **0.0505 (±0.0051)** |

### 5-4. emb_dim=64에서의 최종 비교

| 모델 | emb_dim=32 | emb_dim=64 |
|---|---|---|
| tri (페르소나 있음) | 0.0453 (±0.0022) | 0.0505 (±0.0051) |
| bipartite (페르소나 없음) | 0.0398 (±0.0016) | 0.0496 (±0.0026) |
| **격차** | **tri +14%** | **거의 동률 (차이 0.0009, 양쪽 표준편차보다 작음)** |

두 모델의 emb_dim=64 값 범위가 겹친다(tri: 0.0451~0.0553, bipartite: 0.0478~0.0526).

**최종 해석**: emb_dim=32라는 특정 설정에서만 tri가 유리했고, 둘 다 더 큰 용량(emb_dim=64)을 주면 그 차이가 통계적으로 사라진다. "페르소나가 그래프 임베딩에 확실히 도움된다"고 강하게 주장하기 어렵다 — 있어도 작고, 하이퍼파라미터 선택에 따라 나타났다 사라졌다 하는 불안정한 효과다. "페르소나가 방해된다"는 것도 아니다(emb_dim=64에서 tri가 근소하게, 통계적으로 무의미한 수준으로 앞선다).

## 방법론적 제약 및 한계

- **negative sampling에 seed 없음**: `#37` 5-4에서 확인했듯 run마다 HR@20이 흔들린다 — 그래서 tri와 동일하게 bipartite도 3회 반복해서 평균으로 비교했다. bipartite의 표준편차(±0.0016)는 tri(±0.0022)와 비슷한 수준으로, 관측된 격차(+14%)가 이 노이즈보다 뚜렷하게 크다.
- **페르소나(세그먼트)만 통제된 비교**: 그래프 구조와 하이퍼파라미터는 완전히 동일, u2p/t2p 서브그래프 유무만 다르다 — 이 리포트가 유일하게 페르소나의 순수 효과를 보여주는 비교다.
- **n=3의 한계**: 통계적으로 엄밀한 유의성 검정을 하기엔 반복 횟수가 적다. emb_dim=32에서는 범위가 안 겹쳐서 방향성을 신뢰할 만했지만, emb_dim=64에서는 범위가 겹쳐서 n=3으로는 우열을 가릴 수 없었다 — 더 확실히 하려면 반복 횟수를 늘려야 한다.
- **ALS 비교는 여전히 confound 있음**: 참고용으로만 병기, 페르소나 효과의 근거로 쓰지 않는다.
- **하이퍼파라미터 민감도가 결론을 좌우함(해결됨, 5번 참고)**: `#37`에서 tri에 맞춰 튜닝한 설정(emb_dim=32)을 bipartite에도 그대로 적용했더니 tri가 유리했는데, 둘 다 emb_dim=64로 바꾸니 차이가 사라졌다 — "페르소나 효과"라고 봤던 게 실은 "emb_dim=32가 tri 구조에 우연히 더 잘 맞았던 것"이었을 가능성이 크다. emb_dim 외의 다른 하이퍼파라미터(layer, lr, lamda 등)에서도 비슷한 민감도가 있을 수 있으나 전부 재검증하진 않았다 — 이번 결론(효과 있어도 작고 불안정함)은 emb_dim 축에서만 확인한 것이다.

## 관련 산출물

- `src/baselines/lgcn3/read_data.py` (수정 — `graph_mode` 파라미터 추가, `_resolve_graph_paths()` 순수 함수 분리, TDD 4건)
- `src/baselines/lgcn3/run_lightgcn.py` (수정 — `--graph-mode bipartite` 지원 추가)
- `tests/test_read_data.py` (신규, 4건)
- `data/outputs/LightGCN/hp_tuning_experiment/bipartite_variance_check/` (emb_dim=32 3회 반복 결과 백업, gitignore 대상)
- `data/outputs/LightGCN/hp_tuning_experiment/bipartite_robustness_check/` (emb16/32(layer1)/64 1회씩 백업, gitignore 대상)
- `data/outputs/LightGCN/hp_tuning_experiment/bipartite_emb64_variance/`, `tri_emb64_variance/` (emb_dim=64 3회 반복 결과 백업, gitignore 대상)
- 각 실험의 정확한 조건·소요시간·명령어는 `docs/LIGHTGCN_BIPARTITE_VS_TRI_EXPERIMENT_LOG.md` 참고

## 다음 단계

- [x] ~~bipartite 그래프 로딩 지원 구현~~
- [x] ~~bipartite 3회 반복 실행 및 평가~~
- [x] ~~로버스트니스 체크(하이퍼파라미터 민감도 검증)~~ — emb_dim=32에서만 나타나는 불안정한 효과임을 확인
- [ ] `#34` 이슈에 최종 결과 공유 (결론이 "페르소나가 도움된다"에서 "효과 있어도 작고 불안정하다"로 바뀌었음을 명확히 전달)
- [ ] 대시보드 Tab2("페르소나 기여도") 근거 데이터로 전달 — 다만 결론이 처음 기대(뚜렷한 개선)와 다르다는 점 사전 공유 필요
- [ ] (선택) layer/lr/lamda 등 다른 하이퍼파라미터 축에서도 같은 민감도가 있는지 확인
- [ ] (선택) Twiddler reranking을 ALS/bipartite 위에 적용해 "재랭킹을 통한 페르소나 효과"도 별도로 검증 — 그래프 임베딩 효과가 약하다는 게 확인된 만큼 우선순위가 높아짐
