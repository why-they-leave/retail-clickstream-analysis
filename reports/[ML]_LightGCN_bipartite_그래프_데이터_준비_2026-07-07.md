# [ML] LightGCN bipartite 그래프 데이터 준비 — 2026-07-07

## 분석 목적

Issue #35. #34(LightGCN bipartite vs tri 학습·평가 비교, 페르소나 효과 분리 검증)의 데이터 준비 단계로, `src/datasets/make_lgcn_graph.py`에 u2t만 쓰는 bipartite(페르소나 미결합) 그래프 생성 모드를 추가한다. tri 산출물(#29)을 덮어쓰지 않고 동시에 보존하는 것을 핵심 제약으로 삼았다.

## 데이터셋 버전 및 기간

- #29와 동일: `data/raw/customers.csv`(20,000명), `data/raw/products.csv`(1,197개), `data/interim/sessions_events_products.csv`, `data/interim/orders_items_products.csv`
- 분할 기준: `timestamp < 2025-08-01` = train, 그 이후 = valid (ALS `configs/ALS/params.yaml`의 `split_date`와 동일)
- bipartite 모드는 세그먼트 데이터(`customer_segments_labeled_train_only.csv`)를 아예 읽지 않는다 — u2p/t2p 자체가 없기 때문

## 방법론

```text
customers.csv / products.csv → build_id_encoding()  (tri와 공용, 동일 인덱스 체계)
sessions_events_products.csv / orders_items_products.csv
→ CUTOFF_DATE(2025-08-01) 기준 train/valid 분할
→ build_u2t_mapping()  (train/valid)   : tri와 완전히 동일한 로직·결과 — 모드 무관
→ build_empty_mapping(user_num), build_empty_mapping(item_num)  : u2p/t2p 대신 빈 매핑
→ bipartite_graph_uidx2pidx.json, bipartite_graph_tidx2pidx.json
```

관련 코드: `src/datasets/make_lgcn_graph.py`(수정 — `mode` 파라미터/`--mode` CLI 추가, `build_empty_mapping()` 신규), `tests/test_lgcn_graph.py`(수정 — `TestBuildEmptyMapping` 추가)

**핵심 설계 결정:**

1. **u2t는 tri와 완전히 공유한다.** 같은 `event_types`, 같은 `CUTOFF_DATE`, 같은 인코딩을 쓰므로 모드에 따라 값이 달라질 이유가 없다. 새 로직을 만들지 않고 기존 `build_u2t_mapping()`을 그대로 재사용한다.
2. **u2p/t2p는 tri와 다른 파일 경로(`bipartite_graph_*.json`)에 저장한다.** 기존 `tri_graph_uidx2pidx.json`/`tidx2pidx.json`을 덮어쓰는 방식도 검토했으나, 그러면 bipartite를 한 번 생성한 뒤 tri 학습을 다시 돌리려 할 때 tri 산출물이 사라져 재실행해야 하고, "지금 이 파일이 어느 모드 산출물인지" 혼동할 위험이 있어 기각했다 (`docs/LIGHTGCN_BIPARTITE_DESIGN.md` 참고).
3. **이 이슈는 순수 Python 구조 검증까지만 책임진다.** `read_data.py`/`dense2sparse.py`는 모듈 최상단에서 `import tensorflow`를 하기 때문에, 이 신규 파일이 실제 학습 파이프라인에서 파싱되는지 확인하는 건 tensorflow가 필요하다 — #34/#30에서 다룬다.

## 평가 지표

지도학습 평가가 아니므로, 산출물 무결성과 그래프 통계로 확인했다.

| 지표 | 값 |
|------|-----|
| user_num / item_num | 20,000 / 1,197 (tri와 동일) |
| bipartite u2p — 키 개수 / 전부 빈 리스트 여부 | 20,000개 / True |
| bipartite t2p — 키 개수 / 전부 빈 리스트 여부 | 1,197개 / True |
| tri 산출물(`tri_graph_uidx2pidx.json`, `tri_graph_tidx2pidx.json`) 보존 여부 | 보존됨 (mtime 변경 없음, bipartite 실행이 건드리지 않음) |
| 단위 테스트 | 11/11 통과 (`tests/test_lgcn_graph.py`, 신규 2건 포함) |
| 전체 회귀 테스트 | 30/30 통과 |
| ruff lint/format | 통과 |

## 분석 결과

**핵심 발견:**
- bipartite 모드의 u2p/t2p 키 개수(20,000 / 1,197)가 tri 모드와 정확히 일치한다 — `read_data.py`가 딕셔너리 길이로 `user_num`/`item_num`을 판단하는 구조이므로, 값이 비어 있어도 키 개수가 안 맞으면 조용히 잘못된 그래프 크기가 나올 위험이 있었는데 이 부분을 검증했다.
- bipartite 실행 후에도 tri 산출물(`tri_graph_uidx2pidx.json`, `tri_graph_tidx2pidx.json`)의 mtime이 그대로였다 — 별도 경로 저장 설계가 의도대로 동작함을 확인했다.
- u2t 계산은 모드와 무관하게 동일 로직을 타므로, bipartite 실행 시 세그먼트 CSV를 아예 읽지 않아 #29 대비 실행이 더 가볍다 (lift 계산 없음).

**해석:** 그래프 데이터 준비 자체는 계획대로 가볍게 끝났다. 실제로 시간이 걸리는 부분(학습 인프라 구축, 학습을 두 번 도는 것)은 이 이슈 범위 밖이며, #30/#34에서 팀 시간 여유에 따라 진행 여부가 결정된다.

## 방법론적 제약 및 한계

- `bipartite_graph_*.json`이 `read_data.py`에서 실제로 파싱되는지는 검증하지 못했다 (tensorflow 미설치). #34에서 `read_data.py`가 이 파일 경로를 읽도록 연동하는 작업이 별도로 필요하다.
- bipartite 모드는 u2t 계산을 tri와 별개로 재실행한다 (파일 내용은 동일하지만 매번 다시 계산·저장함). 성능상 문제될 수준은 아니지만, 완전한 캐싱을 원하면 추후 개선 여지가 있다.

## 관련 산출물

- `src/datasets/make_lgcn_graph.py` (수정 — `mode` 파라미터, `--mode` CLI, `build_empty_mapping()` 신규)
- `tests/test_lgcn_graph.py` (수정 — `TestBuildEmptyMapping` 2건 추가, 총 11건)
- `docs/LIGHTGCN_BIPARTITE_DESIGN.md` (신규 — 설계 문서)
- `data/processed/bipartite_graph_uidx2pidx.json`, `bipartite_graph_tidx2pidx.json` (gitignore 대상 로컬 산출물)

## 권장 다음 단계

- [ ] #34: `read_data.py`/`dense2sparse.py`가 `bipartite_graph_*.json`을 읽도록 연동 (tensorflow 설치 후, #30과 함께 검증)
- [ ] #30: `run_lightgcn.py`에 `--graph-mode {tri,bipartite}` 옵션 추가
- [ ] #34: 시간 여유가 되면 bipartite vs tri 학습·평가 비교 실행
