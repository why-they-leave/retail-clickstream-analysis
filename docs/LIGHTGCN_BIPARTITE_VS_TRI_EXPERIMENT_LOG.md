# LightGCN bipartite vs tri — 실험 로그 (Issue #34)

`reports/[ML]_LightGCN_bipartite_vs_tri_페르소나_효과_검증_2026-07-08.md`의 결론을 뒷받침하는 모든 실험의 정확한 조건·명령어·소요시간·결과를 기록한다. 재현하거나 결과를 검증할 때 참고.

공통 조건(따로 표기 없으면 전부 동일):
- 그래프: u2t=구매만(`tri_graph_uidx2tidx_train.json`, 56,230 edge), split_date=2025-08-01
- 고정 하이퍼파라미터: `lamda=0.02`, `neg_samples=10`, `optimizer=Adam(opt=3)`, `batch=10000`, `layer_weight_scheme=uniform`, `lr=0.002`, `epoch=300`
- 평가: `evaluate_lightgcn.py --k 5 10 20`, 평가 유저 1,465명
- 환경: macOS 로컬, TensorFlow 1.x compat 모드, seed=42(임베딩 초기화만 — negative sampling은 비고정)

## 1. bipartite 그래프 로딩 지원 구현 + 스모크 테스트

- 날짜: 2026-07-08
- 작업: `read_data.py`에 `graph_mode` 파라미터 추가(`_resolve_graph_paths()` TDD), `run_lightgcn.py --graph-mode bipartite` 지원 추가
- 스모크 테스트: `python3 run_lightgcn.py --graph-mode bipartite --epoch 2` — 에러 없이 완료, persona_num=0 확인
- 소요시간: 수 초 (2 epoch)

## 2. bipartite 전체 학습 — emb_dim=32(tri-tuned 기본값), 1회차

| 항목 | 값 |
|---|---|
| 명령어 | `python3 run_lightgcn.py --graph-mode bipartite --epoch 300` |
| emb_dim | 32 (params.yaml 기본값) |
| layer | 2 (기본값) |
| 완료 시각 | 2026-07-08 16:22 |
| 학습 소요시간 | 2분 58초 |
| HR@20 | 0.0416 |
| 결과 파일 | `data/outputs/LightGCN/hp_tuning_experiment/bipartite_variance_check/eval_results_bipartite_run1.csv` |

## 3. bipartite 전체 학습 — emb_dim=32, 2·3회차 (재현성 검증용 반복)

| 항목 | run2 | run3 |
|---|---|---|
| 명령어 | `python3 run_lightgcn.py --graph-mode bipartite --epoch 300` (동일 명령 반복) | 〃 |
| 완료 시각 | 2026-07-08 16:54(스크립트 전체 완료 시각) | 〃 |
| 학습 소요시간 | 2분 56초 | 3분 2초 |
| HR@20 | 0.0389 | 0.0389 |
| 결과 파일 | `.../bipartite_variance_check/eval_results_bipartite_run2.csv` | `.../eval_results_bipartite_run3.csv` |

**emb_dim=32, 3회 평균: HR@20 = 0.0398 (±0.0016)**

## 4. bipartite 로버스트니스 체크 — 대체 하이퍼파라미터 1회씩

스크립트: `bipartite_robustness_check.sh`, 완료 시각 2026-07-08 17:11

| 조합 | 명령어 | 학습 소요시간 | HR@20 |
|---|---|---|---|
| emb16, layer2 | `run_lightgcn.py --graph-mode bipartite --epoch 300 --emb-dim 16 --layer 2` | 2분 22초 | 0.0416 |
| emb32, layer1 | `run_lightgcn.py --graph-mode bipartite --epoch 300 --emb-dim 32 --layer 1` | 2분 46초 | 0.0280 |
| **emb64, layer2** | `run_lightgcn.py --graph-mode bipartite --epoch 300 --emb-dim 64 --layer 2` | 4분 6초 | **0.0485** |

## 5. bipartite emb_dim=64 — 2·3회차 (재현성 검증용 반복)

스크립트: `bipartite_emb64_variance.sh`, 완료 시각 2026-07-08 17:22 (run1은 4번 항목의 emb64 결과를 재사용)

| 항목 | run1(4번 재사용) | run2 | run3 |
|---|---|---|---|
| 명령어 | `--graph-mode bipartite --epoch 300 --emb-dim 64 --layer 2` | 〃 | 〃 |
| 학습 소요시간 | 4분 6초 | 3분 55초 | 4분 3초 |
| HR@20 | 0.0485 | 0.0478 | 0.0526 |

**bipartite emb_dim=64, 3회 평균: HR@20 = 0.0496 (±0.0026)**

## 6. tri emb_dim=64 — 1·2·3회차 (공정 비교를 위한 대조 실험)

스크립트: `tri_emb64_variance.sh`, 완료 시각 2026-07-08 17:42

| 항목 | run1 | run2 | run3 |
|---|---|---|---|
| 명령어 | `--graph-mode tri --epoch 300 --emb-dim 64 --layer 2` | 〃 | 〃 |
| 학습 소요시간 | 4분 17초 | 4분 23초 | 4분 24초 |
| HR@20 | 0.0451 | 0.0553 | 0.0512 |

**tri emb_dim=64, 3회 평균: HR@20 = 0.0505 (±0.0051)**

## 요약표

| 실험 | emb_dim | layer | HR@20 (3회 평균 또는 1회) | 반복 표준편차 |
|---|---|---|---|---|
| tri, tri-tuned (참고, #37에서 확정) | 32 | 2 | 0.0453 | ±0.0022 |
| bipartite, tri-tuned | 32 | 2 | 0.0398 | ±0.0016 |
| bipartite | 16 | 2 | 0.0416 (1회) | — |
| bipartite | 32 | 1 | 0.0280 (1회) | — |
| bipartite | 64 | 2 | **0.0496** | ±0.0026 |
| tri | 64 | 2 | **0.0505** | ±0.0051 |

## 소요시간 총합

전체 실험(1~6번)에 걸린 학습 시간만 합산하면 약 43분 (1회차 스모크 테스트, 평가 시간 제외). 그래프 로딩·평가·리포트 작성 등을 포함한 전체 작업 시간은 이보다 길다.

## 관련 문서

- 결론: `reports/[ML]_LightGCN_bipartite_vs_tri_페르소나_효과_검증_2026-07-08.md`
- tri 쪽 확정 하이퍼파라미터 근거: `reports/[ML]_LightGCN_tri_전체학습_및_ALS_비교평가_2026-07-08.md` (5차 실행)
