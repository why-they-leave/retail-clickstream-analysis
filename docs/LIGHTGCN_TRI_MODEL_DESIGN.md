# LightGCN_tri Model Design

## 목적

Issue #30(LightGCN 모델 실행 환경 구축·평가)을 진행하려면 `model_LightGCN_tri` 클래스가 필요한데, 확인 결과 **이 클래스는 코드베이스에 존재하지 않는다**. `src/baselines/lgcn3/train_model.py`가 `MODEL == 'LightGCN_tri'`일 때 이 클래스를 호출하지만 어디에도 정의·임포트돼 있지 않다. 이 문서는 이 클래스를 어떻게 새로 작성할지 설계한다.

## 배경 — 왜 없던 걸 알아차리는 데 오래 걸렸나

`src/baselines/lgcn3/`에는 모델 클래스가 두 개만 있다.

| 클래스 | 파일 | 필요 입력 |
|---|---|---|
| `model_LGCN_tri` | `model_LGCN_tri.py` | **사전학습된 frequency embedding** (`pretraining/_graph_embeddings_tri.py`로 생성, 고유값 분해 필요) |
| `model_LGCN_afd_tri` | `model_LGCN_afd_tri.py` | 위와 동일 + AFD(상관관계 정규화) |

`LightGCN_tri`(원 논문 방식, 그래프를 즉석에서 구성해 바로 전파하는 방식)는 #28에서 "실제로 구현 가능한 타깃"이라고 적혀 있었지만, 실제로는 판단만 됐지 코드가 없는 상태였다.

## 설계 결정: A안(신규 구현) vs B안(LGCN_tri 사전학습 경로) — A안 채택

| | A. `model_LightGCN_tri` 신규 구현 | B. 기존 `model_LGCN_tri` + 사전학습 |
|---|---|---|
| 사전 단계 | 불필요 | `pretraining/_graph_embeddings_tri.py`로 21,203×21,203 라플라시안 고유값 분해 필요 — `which='SM'`(shift-invert 없음)이라 수렴 시간/성공 여부 불확실 |
| 우리 데이터 반영 | `dense2sparse.py`의 `propagation_matrix_tri()`(이미 구현·테스트됨, lift 가중치 반영됨)를 그대로 입력으로 씀 | 사전학습 스크립트가 이진 엣지(`A[i,j]=1`)만 지원 — lift 가중치가 사전학습 단계에서 소실됨. `persona_number=20` 하드코딩도 6으로 고쳐야 함 |
| 하이퍼파라미터 | 표준 LightGCN이라 선택지가 적음(레이어 수, 임베딩 차원, lr, lamda 정도) | `graph_conv`/`prediction`/`generalization`/`if_transformation`/`activation`/`pooling` 등 문서 없는 아키텍처 선택지를 추측해야 함 |
| 코드 작업량 | `model_LGCN_tri.py`의 "그래프 전파" 블록만 교체, 나머지(placeholder/loss/optimizer)는 재사용 | 사전학습 스크립트 재작성 + 기존 모델 재사용 |

**결론: A안.** 표준 LightGCN 알고리즘이 원래 spectral 방식(B)보다 단순해서, 새로 짜야 하는 부분이 오히려 더 적다.

## `model_LGCN_tri.py` 대비 무엇을 재사용하고 무엇을 바꾸는가

`model_LGCN_tri.py`(187줄)를 줄 단위로 검토한 결과:

**그대로 재사용 가능 (약 70%)**
- placeholder 정의 (`users`, `pos_items`, `neg_items`, `items_in_train_data`, `top_k`) — 16~43행
- 임베딩 변수 초기화 (`user_embeddings`, `item_embeddings`, `persona_embeddings`, 랜덤 초기화) — 49~53행
- `pooling == 'Sum'` 레이어 가중치 로직(`1/(l+1)`, 표준 LightGCN의 레이어 평균과 동일) — 57~58행
- 예측 방식 `InnerProduct` (표준 LightGCN이 쓰는 방식) — 107~112행
- BPR loss, regularization, optimizer, `var_list`/`updates` — 130~150행, 152~186행 (`bpr_loss`, `regularization` 함수 그대로)

**바꿔야 하는 부분 (그래프 전파 블록, 74~89행)**

기존(spectral, frequency embedding 기반):
```python
self.embeddings = tf.matmul(
    tf.matmul(self.graph_emb, tf.linalg.tensor_diag(self.kernel[l])),
    tf.matmul(self.graph_emb, self.embeddings, transpose_a=True, transpose_b=False),
)
```

신규(표준 LightGCN, `propagation_matrix_tri()`가 만든 sparse 인접행렬을 그대로 곱함 — 레이어마다 학습되는 `kernel` 자체가 없는 게 표준 LightGCN의 핵심 단순화):
```python
self.embeddings = tf.sparse.sparse_dense_matmul(self.sparse_graph, self.embeddings)
```

**아예 빼는 부분**
- `kernel`/`frequency`/`graph_conv` 분기 — spectral 전용, 표준 LightGCN엔 없음
- `if_transformation`(레이어 간 변환 행렬) — 표준 LightGCN은 안 씀
- `activation`(Sigmoid/Tanh/ReLU) — 표준 LightGCN은 레이어 간 비선형 활성화를 안 씀 (선형 전파가 핵심)
- `if_pretrain` 분기 — 우리는 pretrained latent factor가 없음, 랜덤 초기화 경로만 사용

## 예상 시그니처

```python
class model_LightGCN_tri(object):
    def __init__(self, n_users, n_items, n_personas, lr, lamda, emb_dim, layer,
                 sparse_graph, optimization):
        ...
```

`model_LGCN_tri`의 시그니처(`graph_embeddings, graph_conv, prediction, loss_function, generalization, if_pretrain, if_transformation, activation, pooling`)보다 훨씬 짧다 — 표준 LightGCN이라 선택할 게 적기 때문. `train_model.py`의 `if MODEL == 'LightGCN_tri': model = model_LightGCN_tri(...)` 호출부(37행)도 이 짧아진 시그니처에 맞게 같이 고쳐야 한다.

## 통합 지점

- 입력: `dense2sparse.py`의 `propagation_matrix_tri(graph, user_num, item_num, persona_num, norm='sym_norm')`가 반환하는 `tf.SparseTensor` — u2t/u2p/t2p 세 그래프를 이미 하나의 정규화된 인접행렬로 합쳐놓은 것. #35(bipartite)에서도 이 함수를 u2p/t2p 빈 매핑으로 그대로 재사용할 수 있는지가 #34의 검증 과제.
- `read_data.py`가 이 JSON들을 로드해 `(uidx, tidx)`/`(uidx, pidx)`/`(tidx, pidx, weight)` 튜플 리스트로 변환한 뒤 `propagation_matrix_tri()`에 넘기는 구조 — 이 연동 자체(#34에서 확인 예정)와 이번 모델 클래스 작성은 별개 작업이지만 순서상 모델 클래스가 먼저 있어야 end-to-end 스모크 테스트가 가능하다.

## 미해결 질문 / 리스크

- 표준 LightGCN 논문은 레이어 수 K를 보통 2~3으로 씀 — `configs/`에 하이퍼파라미터를 어디까지 노출할지(#30에서 CLI 설계 시 결정)
- `n_personas`가 tri 모드에선 6(세그먼트 수), bipartite 모드에선 0에 가까운 상태(#35 참고, 노드는 만들되 엣지 없음) — `model_LightGCN_tri`가 `n_personas=0`을 받아도 안전하게 동작하는지 확인 필요 (persona_embeddings shape가 `[0, emb_dim]`이 되는 경우)
- 이 클래스에 대한 단위 테스트는 TensorFlow 그래프 실행이 필요해 순수 함수 테스트보다 무겁다 — 최소한 "그래프가 에러 없이 빌드되는지"(스모크 테스트) 수준부터 시작하는 게 현실적

## 관련 문서

- 모델 클래스가 없다는 것을 발견한 배경: 이슈 #30, #28
- 입력 그래프 데이터: `docs/LIGHTGCN_BIPARTITE_DESIGN.md`, `reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`
