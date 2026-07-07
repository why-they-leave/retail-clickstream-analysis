# [ML] LightGCN_tri 모델 구현 및 스모크테스트 — 2026-07-07

## 분석 목적

Issue #30. `docs/LIGHTGCN_TRI_MODEL_DESIGN.md`에서 설계한 `model_LightGCN_tri`(표준 LightGCN 전파, #28에서 "실제로 구현 가능한 타깃"이라고만 판단되고 코드는 없던 모델)를 실제로 구현하고, #29 산출물(실제 tri-graph 데이터)로 학습이 에러 없이 도는지, epoch당 시간이 얼마나 걸리는지 확인한다. 성능(F1/NDCG) 검증이 아니라 "파이프라인이 실제로 작동하는지" 확인이 목적이다.

## 데이터셋 버전 및 기간

- #29 산출물 그대로 사용: `data/processed/tri_graph_uidx2tidx_{train,valid}.json`, `tri_graph_{uidx2pidx,tidx2pidx}.json`
- user_num=20,000, item_num=1,197, persona_num=6 (v2 6-segment)

## 방법론

```text
1. model_LightGCN_tri 클래스 구현 (TDD)
   → 더미 데이터(항등행렬, 유저 3~4명)로 단위 테스트 5건 작성·통과
2. train_model.py/read_data.py 연동 수정
   → model_LightGCN_tri를 새 시그니처로 호출하도록 정리
3. 실제 데이터로 스모크 테스트
   → --model 11(LightGCN_tri) --dataset 2(MBA) --epoch 2 로 실행
   → 여기서 기존 legacy 코드의 버그 4개를 추가로 발견·수정 (아래 참고)
```

## 평가 지표

지도학습 성능 평가가 아니라 파이프라인 무결성 확인이 목적이므로, 아래를 지표로 삼았다.

| 지표 | 값 |
|------|-----|
| 에러 없이 완주 여부 | 완주 (2 epoch) |
| 단위 테스트 | 5/5 통과 (신규), 전체 회귀 35/35 통과 |
| epoch당 소요 시간 | 약 12.7초 |
| 2 epoch 총 소요 시간 | 25.5초 |

## 분석 결과

**핵심 발견:**

- **model_LightGCN_tri가 실제 20,000명×1,197개×6세그먼트 규모 데이터에서 정상 동작함**을 확인했다 — 지금까지는 항등행렬 기반 더미 데이터로만 검증했었는데, 실제 규모에서도 차원/메모리 문제 없이 돌았다.
- **epoch당 약 12.7초** — 원래 기본값인 300 epoch로 전체 학습을 돌리면 대략 1시간 안팎(300 × 12.7초 ≈ 63분)이 소요될 것으로 추정된다. 이 수치가 #34(bipartite vs tri 비교, "시간 되면 진행") 진행 여부를 판단하는 근거가 된다 — tri 학습 1회에 약 1시간이면, bipartite까지 포함해 2회 학습에는 약 2시간이 필요하다는 뜻이다.
- **legacy 코드에 실행을 막는 버그가 4개 더 있었다** (모두 #30 스모크 테스트 전에는 아무도 실제로 실행해본 적이 없어서 발견 못 했던 것들, #28의 "코드가 실행 가능한 상태가 아니다"라는 진단과 일치):

  | # | 파일 | 버그 | 수정 |
  |---|---|---|---|
  | 1 | `read_data.py` | `persona_num=20` 하드코딩 (v1 20-persona 시절 값) | `6`으로 수정 (v2 6-segment) |
  | 2 | `train_model.py` | `model_LightGCN_tri` 호출부가 `pre_train_latent_factor`/`if_pretrain` 등 옛 시그니처 인자를 넘김 | 새 시그니처(`sparse_graph`, `optimization`만)에 맞게 정리 |
  | 3 | `read_data.py` | `read_all_data_tri()`의 `all_para` 언패킹 리스트가 29개인데 실제 `all_para`는 30개(`AFD_ALPHA` 누락) — `ValueError: too many values to unpack` | 끝에 `_` 하나 추가 |
  | 4 | `run_lgcn3.py` 패턴 | `train_model.train_model(params.all_para[:26], ...)`로 26개만 잘라 넘기는데, 함수 내부는 `para[13:]`에서 17개(총 30개)를 요구 — `ValueError: not enough values to unpack` | 전체 `all_para`(30개)를 그대로 전달 |
  | 5 | `evaluation.py` | `from numpy import *`가 파이썬 내장 `max()`/`len()`/`set()`을 numpy 버전으로 가려서, `max(len(top_k_items), epsilon)`이 `numpy.max(array, axis=epsilon)`로 오해석돼 `TypeError` | 실제로 쓰는 `log2`만 명시적으로 임포트 |

- 그 외 `model_LightGCN_tri` 자체에서 발견한 것: `train_model.py`/`test_model.py`가 모델 종류와 무관하게 `feed_dict`에 `model.keep_prob`를 항상 넣는데, 표준 LightGCN이라 dropout이 없어서 이 placeholder를 빠뜨렸었다 — TDD로 재현 테스트를 먼저 작성해 잡았다.

**해석:** 스모크 테스트가 원래 목적(에러 없이 도는지, 시간이 얼마나 걸리는지) 외에도, 그동안 아무도 실행 안 해본 legacy 코드 경로 전체를 처음으로 검증하는 역할을 했다. F1_max=0은 2 epoch만 돌렸으니 당연한 결과이고 성능 지표로서 의미는 없다.

## 방법론적 제약 및 한계

- 이번 스모크 테스트는 하이퍼파라미터(lr, lamda, layer 등) 전부 `parse.py`의 기본값을 그대로 썼다 — 실제 성능 검증을 위한 튜닝은 하지 않았다.
- epoch당 12.7초는 이 로컬 macOS 환경(GPU 없이 CPU) 기준이다. CI나 다른 머신에서는 다를 수 있다.
- `read_data.py`의 `persona_num` 분기는 `DATASET == 'MBA'`일 때만 6으로 고쳤다. 다른 데이터셋(Instacart 등) 분기는 이번에 안 건드렸다 — 이 프로젝트에서 실제로 쓰는 건 MBA(우리 데이터를 이 이름으로 매핑해서 씀)뿐이라 문제는 없지만, 코드 자체는 여전히 다른 프로젝트에서 옮겨온 흔적이 남아있다.
- bipartite 모드(#35)와의 실제 연동(`read_data.py`가 `bipartite_graph_*.json`을 읽게 하는 것)은 이번 스모크 테스트 범위 밖이다 — #34에서 확인 예정.

## 관련 산출물

- `src/baselines/lgcn3/model_LightGCN_tri.py` (신규)
- `tests/test_model_lightgcn_tri.py` (신규, 5건)
- `tests/conftest.py` (신규 — pandas/tensorflow import 순서 데드락 회피)
- `src/baselines/lgcn3/read_data.py`, `train_model.py`, `evaluation.py` (수정 — 위 표의 버그 4개)
- `pyproject.toml`/`uv.lock` (수정 — `tensorflow`, `openpyxl` 의존성 추가)
- `docs/LIGHTGCN_TRI_MODEL_DESIGN.md` (설계 문서, 사전 작성)

## 권장 다음 단계

- [ ] `run_lightgcn.py` 신규 작성 — ALS 스타일 CLI 진입점 (`--graph-mode {tri,bipartite}` 확장 여지 포함)
- [ ] 학습 결과를 CSV로 저장하는 로직 추가 (현재는 엑셀/print만 함)
- [ ] `evaluate_lightgcn.py` 신규 작성 — Hit Ratio@20 추가
- [ ] 전체 epoch(300, 약 1시간 추정)로 실제 학습 실행 및 성능 리포트 작성
- [ ] #34: 시간 여유가 되면 bipartite 산출물(#35)로도 같은 스모크 테스트 진행
