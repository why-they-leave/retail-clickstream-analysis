# LightGCN Bipartite Graph Design

## 목적

Issue #35에서는 LightGCN bipartite(u2t만, 페르소나 미결합) 그래프 데이터를 준비한다. 이 산출물은 #34(학습·평가 비교)에서 tri-graph(u2t+u2p+t2p, 페르소나 결합)와 같은 조건으로 비교해, "페르소나 효과"를 "모델 종류 효과"와 분리하기 위해 쓰인다. 배경은 `docs/WHY_SEGMENTS.md`, 설계 배경 논의는 #28/#34 참고.

```text
[tri, #29에서 이미 구축]        [bipartite, #35]

  유저 ──┬── 상품               유저 ────── 상품
         └── 세그먼트                        (u2p/t2p 없음)
  상품 ────── 세그먼트
```

## 전체 흐름

`src/datasets/make_lgcn_graph.py`의 `main(mode="tri" | "bipartite")`로 분기한다.

```text
customers.csv / products.csv
→ build_id_encoding()                         : tri와 동일 (공용)
sessions_events_products.csv / orders_items_products.csv
→ CUTOFF_DATE(2025-08-01) 기준 train/valid 분할 : tri와 동일
→ build_u2t_mapping()  (train/valid)          : tri와 완전히 동일한 로직/결과 — 모드 무관, 재계산만 다시 함
→ (bipartite 모드) u2p/t2p는 계산하지 않고 전부 빈 리스트로 채움
→ (tri 모드) 기존과 동일하게 build_u2p_mapping() / build_t2p_mapping() 실행
```

u2t는 모드와 무관하게 100% 동일한 값이 나온다 (같은 `event_types`, 같은 `CUTOFF_DATE`, 같은 인코딩을 쓰기 때문). 그래서 bipartite 모드에서도 u2t 계산 자체는 새로 작성하지 않고 기존 함수를 그대로 호출한다.

## 파일 경로 설계 결정

**문제**: u2p/t2p 산출물을 tri 모드와 bipartite 모드가 같은 파일 경로(`tri_graph_uidx2pidx.json`, `tri_graph_tidx2pidx.json`)에 쓰면, 나중에 모드를 바꿔 실행할 때마다 이전 모드의 산출물이 사라진다. #34에서 tri와 bipartite를 같은 조건으로 비교하려면 두 산출물이 동시에 존재해야 하므로 이 방식은 위험하다.

**결정**: u2p/t2p는 모드별로 다른 파일명을 쓴다. u2t는 모드 무관하게 동일한 값이므로 기존 경로를 공유한다.

| 그래프 | tri 모드 | bipartite 모드 |
|--------|----------|-----------------|
| u2t (train) | `data/processed/tri_graph_uidx2tidx_train.json` (공유) | 동일 |
| u2t (valid) | `data/processed/tri_graph_uidx2tidx_valid.json` (공유) | 동일 |
| u2p | `data/processed/tri_graph_uidx2pidx.json` | `data/processed/bipartite_graph_uidx2pidx.json` (신규, 전부 빈 리스트) |
| t2p | `data/processed/tri_graph_tidx2pidx.json` | `data/processed/bipartite_graph_tidx2pidx.json` (신규, 전부 빈 리스트) |

이렇게 하면 tri/bipartite 산출물이 동시에 디스크에 존재할 수 있어, #34에서 모드를 바꿔가며 재생성할 필요 없이 바로 학습·평가를 전환할 수 있다.

**대안(기각)**: 기존 `tri_graph_uidx2pidx.json`/`tidx2pidx.json`을 모드에 따라 덮어쓰는 방식도 검토했다. `read_data.py` 수정이 필요 없다는 장점은 있지만, bipartite로 한 번 실행하면 tri 산출물이 사라져 재실행해야 하고 "지금 이 파일이 어느 모드 산출물인지" 혼동할 위험이 있어 기각했다.

## 검증 기준 (#29와 동일)

- u2t/u2p/t2p 모두 `read_data.py`가 요구하는 형식(모든 유저/상품이 키로 존재, 값은 빈 리스트 허용)을 만족해야 한다 — `user_num`/`item_num`이 딕셔너리 길이로 결정되는 구조이기 때문
- bipartite 모드의 u2p/t2p는 값이 전부 빈 리스트여야 하지만, **키 개수(user_num/item_num)는 tri와 동일**해야 한다

## 실행 방법

```bash
# 레포 루트에서 실행 (src.-prefixed 임포트, editable install 필요: pip install -e ".[dev]")
python3 -m src.datasets.make_lgcn_graph --mode bipartite
#   출력: data/processed/tri_graph_uidx2tidx_{train,valid}.json (tri와 공유)
#         data/processed/bipartite_graph_{uidx2pidx,tidx2pidx}.json (신규, 전부 빈 값)

# tri 모드(기존, #29)는 --mode 생략 시 기본값
python3 -m src.datasets.make_lgcn_graph
# 또는 명시: python3 -m src.datasets.make_lgcn_graph --mode tri
```

## 이번 이슈(#35)의 범위와 한계

- 이 이슈는 JSON 산출물을 만들고 구조적으로 검증(순수 Python, tensorflow 불필요)하는 데까지만 책임진다.
- `read_data.py`/`dense2sparse.py`는 모듈 최상단에서 `import tensorflow as tf`를 하기 때문에, 이 신규 파일명(`bipartite_graph_*.json`)을 실제 학습 파이프라인이 읽게 하려면 `read_data.py`에 파일 경로 매핑(또는 모드 인자)을 추가하는 작업이 별도로 필요하다 — 이건 #34/#30 몫이다.
- t2p 계산(`build_t2p_mapping`)의 lift 임계값·최소 구매 건수 등 tri 쪽 설계는 이 문서에서 다루지 않는다 — `docs/WHY_SEGMENTS.md`, #29 리포트(`reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`) 참고.

## 관련 문서

- 왜 이 비교가 필요한지: `docs/WHY_SEGMENTS.md`, 이슈 #28/#34
- tri-graph 원본 설계: `reports/[ML]_LightGCN용_tri-graph_데이터_파이프라인_구축_2026-07-06.md`
- 실제 검증 결과(테스트/린트 통과 여부, 산출물 무결성): `reports/[ML]_LightGCN_bipartite_그래프_데이터_준비_2026-07-07.md`
