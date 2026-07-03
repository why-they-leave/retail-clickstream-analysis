# Segment Naming v2 — 채택 기록

Issue #17 LLM naming 실험의 run별 채택 여부를 기록한다.

## 진행 방식

- 실행: `PYTHONPATH=src python3.11 src/persona/segment_naming.py --dataset all --n-runs N` (@eric010314-sys)
  - 매 실행 결과는 `experiments/segment_naming_v2/run_<날짜>_<순번>/segment_personas.json`에 저장되며, canonical 파일(`data/processed/segment_personas_v2.json`)은 건드리지 않는다.
- 채택 심사: 아래 기준으로 후보 run을 검토한다 (@JungYeoni)
  - evidence 항목이 실제 `segment_summary_all_customers.csv` 수치와 정확히 일치하는가
  - demographic/lifestyle 표현이 없는가 (`Young`, `Affluent`, `Family`, `Professional` 등)
  - segment_name이 행동 기반 표현으로 구체적인가 (과도하게 broad/marketing 톤 지양)
- 채택 확정: 이 파일에 기록을 남기고, 이슈 #17 코멘트에도 "v2 확정: run_X" 남긴다.
- 반영: `PYTHONPATH=src python3.11 src/persona/segment_naming.py --dataset all --promote run_X` (API 호출 없이 canonical 파일에 복사)

## 기록

| 날짜 | run | 채택 여부 | 사유 | 채택자 |
|---|---|---|---|---|
| 2026-07-03 | run_2026-07-03_1 | 사전 체크 통과, 최종 채택 대기 | `temperature=0`, 10회 반복 중 6개 세그먼트 전부 다수 의견과 일치. evidence 전량 `segment_summary_all_customers.csv` 수치와 기계 대조 완료(불일치 0건), demographic/lifestyle 키워드 0건. 정성 검토(broad/marketing 톤 여부)는 아직 미완료 | (사전 체크: Claude) |
