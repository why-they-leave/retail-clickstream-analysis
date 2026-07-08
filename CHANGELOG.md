# 변경 이력 (Changelog)

## 미출시 변경사항

### add

- #7 ALS 성능 개선 및 threshold 검증 리포트 ([49da580](https://github.com/JungYeoni/da-template/commit/49da58030ef5d5d0989af71f649ee861a9a095b9))
- #7 ALS 하이퍼파리미터 튜닝 및 성능 향상 분석, cold 유저 threshold 실험 ([eb4f116](https://github.com/JungYeoni/da-template/commit/eb4f11690198dfac4be39cea189d0b778d177e19))
- #7 코드 설명 주석 추가 ([1725993](https://github.com/JungYeoni/da-template/commit/1725993d74a74918797931a07891155c1eed4d4b))

### data

- 고객별 세그먼트 배정 전체 테이블 공유 (#30) ([5ef1a68](https://github.com/JungYeoni/da-template/commit/5ef1a68268a179629c43dd5733b262f90282631c))
- Tri-graph uidx/tidx <-> user_id/product_id 매핑 CSV 공유 (#30) ([60b3f81](https://github.com/JungYeoni/da-template/commit/60b3f816b671b69e16803c88924faf9909e0733b))
- LightGCN용 train-only 세그먼트 이름/정의 canonical 공유 (#31) ([9a0c8fd](https://github.com/JungYeoni/da-template/commit/9a0c8fd03278c2f1ffb237570f484b16b4503585))
- V1 GPLR 페르소나 canonical 산출물 공유 (final_personas.json) ([d66df61](https://github.com/JungYeoni/da-template/commit/d66df6197c0743c6530023f902cdcb6c5bc4e70e))

### style

- Ruff format 적용 (#7) ([92070e3](https://github.com/JungYeoni/da-template/commit/92070e3aad89b5b9de267c7b2b1378c22a09a3c6))
- Ruff format 적용 (#17) ([b1e9521](https://github.com/JungYeoni/da-template/commit/b1e9521959b34c855b44cad2c95ecdd70cb2f5c3))
- Ruff import 정렬 수정 #5 ([edd97fe](https://github.com/JungYeoni/da-template/commit/edd97fedd875dd593607c6c78e21e0082224fa60))
- Ruff format 재적용 #1 ([fb19d42](https://github.com/JungYeoni/da-template/commit/fb19d423b8bbf23878774a5ac355678f8056dfac))
- Ruff format 적용 #1 ([cee6110](https://github.com/JungYeoni/da-template/commit/cee6110e42a9a69fe5361adaaee587fbfaa467cd))

### 기타

- CodeRabbit 리뷰 트리거용 빈 커밋 (#15) ([ead6f4a](https://github.com/JungYeoni/da-template/commit/ead6f4a8b1c3847290e402876d00f235579a6295))
- 코덱스 설정 업데이트 ([6d7727b](https://github.com/JungYeoni/da-template/commit/6d7727b6c642d44cb1e7c9f0837dce96787b1975))
- 불필요한 패키지 삭제 및 패키지 추가 (#8) ([cf0c7c6](https://github.com/JungYeoni/da-template/commit/cf0c7c6ea0bed5913289d68b6b054ac3ff1a3b13))
- Gitignore에 로컬 실행 설정 파일 추가 (#8) ([092d24e](https://github.com/JungYeoni/da-template/commit/092d24e447961db2c08540ca7dc341569c030117))
- 코드래빗 한글 설정 (#8) ([4fcea59](https://github.com/JungYeoni/da-template/commit/4fcea5999408a430aeb159b2cc66cd80599afb6c))
- Yaml 파일 설정 프로젝트에 맞게 수정 (#8) ([0bd3943](https://github.com/JungYeoni/da-template/commit/0bd39431a64ee0a96eedca640370531c40ce63be))
- Gitignore에서 /docs 제거 (#5) ([6c5437b](https://github.com/JungYeoni/da-template/commit/6c5437b92a2cb8a88be19e274e70c1db26d7357b))
- 템플릿 설정 동기화 (#5) ([4176e64](https://github.com/JungYeoni/da-template/commit/4176e6487e41dcbbc53e1ae4f7bcf9e635d2f2dc))
- Gitignore에 가상환경 추가 #5 ([7de05a3](https://github.com/JungYeoni/da-template/commit/7de05a30f97e53e28b913c906c64c72773cbe13a))
- 템플릿 설정 동기화 ([7559d2a](https://github.com/JungYeoni/da-template/commit/7559d2ac249dab475f0a7701f2d05c9b21e9592b))
- 기존 템플릿 업데이트 사항 ([d792410](https://github.com/JungYeoni/da-template/commit/d7924109054aa27f8f2a08746f9f97582f26cc02))
- Notebook-smoke-test 제거 및 issue-helper 워크플로우 추가 ([c7d43fc](https://github.com/JungYeoni/da-template/commit/c7d43fc7f3ff3d30445e50adf369cd6b5c9e2048))
- READEME에 데이터 다운로드 방법 추가 ([6ba348b](https://github.com/JungYeoni/da-template/commit/6ba348b4a1643a3b09e1e6cc4d0a80256735fd4e))
- ERD 이미지 추가 ([ab0fac9](https://github.com/JungYeoni/da-template/commit/ab0fac941ba050ef65d948a7cc6010075994298d))
- 프로젝트 초기 세팅 ([94590af](https://github.com/JungYeoni/da-template/commit/94590af1759bdbbf39316d8e574ccabeda4960b3))

### 리팩터링

- #7 ALS 하이퍼파리미터 수정 및 파이프라인 오류, 피드백 반영 ([236b6d4](https://github.com/JungYeoni/da-template/commit/236b6d43d86a5680d8384654de049c05ead90566))
- #7 전반적인 경로 수정 및 데이터마트 생성 로직, 테스트용 GT 수정, 병렬처리 등 포함 - 확정 커밋 아니고, 아직 수정이 필요한 커밋입니다. 노트북에서 작업을 이어가기 위해서 커밋합니다 ([3d30154](https://github.com/JungYeoni/da-template/commit/3d3015472a287773ae1e1e8bf14b52f74b22515d))
- Segment assignment 리뷰 반영 (#16) ([2fce1d3](https://github.com/JungYeoni/da-template/commit/2fce1d31aae62e3069dfc1b5fe95c847818a31e8))

### 문서

- CHANGELOG 자동 업데이트 [skip ci] ([fae0436](https://github.com/JungYeoni/da-template/commit/fae0436c6633ac28c7c7912a178f2a958004fb89))
- CHANGELOG 자동 업데이트 [skip ci] ([d036e57](https://github.com/JungYeoni/da-template/commit/d036e57f8dd25b8e17d489c11cedee0b105e0c6a))
- CHANGELOG 자동 업데이트 [skip ci] ([8fc60fd](https://github.com/JungYeoni/da-template/commit/8fc60fd717f28ce5e78877d0d96d4d27e6abb7ee))
- LightGCN bipartite 그래프 데이터 준비 설계/결과 문서화 (#35) ([37ceb1a](https://github.com/JungYeoni/da-template/commit/37ceb1abec0bf61a2936a59c2131e4763994b615))
- CHANGELOG 자동 업데이트 [skip ci] ([f1c4bc3](https://github.com/JungYeoni/da-template/commit/f1c4bc38657465f6620c7375b5dd2feb5acf7c66))
- CHANGELOG 자동 업데이트 [skip ci] ([acbd92e](https://github.com/JungYeoni/da-template/commit/acbd92ea77e9da065c3896e58a26e63cabe160f4))
- V1/v2 페르소나 canonical 공유 방식 문서화 ([b926ccc](https://github.com/JungYeoni/da-template/commit/b926ccc0dd02dfa1ec67b69f064514608cddcdbd))
- CHANGELOG 자동 업데이트 [skip ci] ([2dc6be3](https://github.com/JungYeoni/da-template/commit/2dc6be372a320f35fa8a2f29294f54e399d6b22c))
- CHANGELOG 자동 업데이트 [skip ci] ([dbcc904](https://github.com/JungYeoni/da-template/commit/dbcc9047376eff7a631ea280b72ae07d63052f30))
- 페르소나/세그먼트 실행 커맨드 문서 추가 (#29) ([c2c1505](https://github.com/JungYeoni/da-template/commit/c2c1505e412248aeede8fa410fddb0df8803c1f6))
- LightGCN tri-graph 데이터 파이프라인 구축 리포트 작성 (#29) ([2a6dfad](https://github.com/JungYeoni/da-template/commit/2a6dfada9200e407dfefdb553d5d0d532b3f7efe))
- CHANGELOG 자동 업데이트 [skip ci] ([f45df5e](https://github.com/JungYeoni/da-template/commit/f45df5e1cf09cebff58ecf9e63d16f7259bab66e))
- 리포트에 실행 방법 및 재실행 시 과금 주의사항 추가 (#31) ([6eb1f6c](https://github.com/JungYeoni/da-template/commit/6eb1f6cba4658233baf4c904d9f859dd03333890))
- LightGCN 운영 시 세그먼트 drift 반영 방식 리포트에 추가 (#31) ([ebe76d6](https://github.com/JungYeoni/da-template/commit/ebe76d69975b42883126d6917ef24d809f6e999d))
- CHANGELOG 자동 업데이트 [skip ci] ([f3068b9](https://github.com/JungYeoni/da-template/commit/f3068b9fb83c48d40e211614a418222a5882b2e1))
- 프로젝트에서 세그먼트를 도입한 배경과 목적 정리 ([dc01432](https://github.com/JungYeoni/da-template/commit/dc01432e8970573876303307509bd4640592f7ff))
- CHANGELOG 자동 업데이트 [skip ci] ([d442d25](https://github.com/JungYeoni/da-template/commit/d442d25302ce50a16e6a7cdec5e46922e1c23fa5))
- 리포트에 단위 테스트 및 import 경로 수정 반영 (#26) ([306c5d1](https://github.com/JungYeoni/da-template/commit/306c5d1d7c15d06ed57b4cc1dd6ca99bb4bdcb0a))
- 고객별 세그먼트 라벨 테이블 산출물 리포트 반영 (#26) ([9c708a9](https://github.com/JungYeoni/da-template/commit/9c708a99ee6ad0619f92db72d918f444d8eba67e))
- CHANGELOG 자동 업데이트 [skip ci] ([655952f](https://github.com/JungYeoni/da-template/commit/655952fd7dd9c4bf555aefbf89933d2be959501b))
- 신규 데이터 유입 시 segment 불안정 가능성 기록 (#18) ([3685567](https://github.com/JungYeoni/da-template/commit/3685567b2d80c0c8fd424feac0b2c4f01ad9575a))
- V2 세그먼트 방식 적합성 설명 추가 (#18) ([b16a3f3](https://github.com/JungYeoni/da-template/commit/b16a3f34ca81e00b1a6551b815881abfa311f519))
- CHANGELOG 자동 업데이트 [skip ci] ([87761da](https://github.com/JungYeoni/da-template/commit/87761dafdf0d656b87f235e6356d026c66cd975f))
- CHANGELOG 자동 업데이트 [skip ci] ([f342eec](https://github.com/JungYeoni/da-template/commit/f342eecefc8bc44e849b6932e83aca0ea1c69121))
- CHANGELOG 자동 업데이트 [skip ci] ([c2c2c67](https://github.com/JungYeoni/da-template/commit/c2c2c671f9abc318781c44056da1cf03f02c99a6))
- CHANGELOG 자동 업데이트 [skip ci] ([bd247cb](https://github.com/JungYeoni/da-template/commit/bd247cbb84ba923bf726bf431aa933576efed99a))
- .env.example 추가 및 로컬 환경 세팅 가이드 보강 ([a35916f](https://github.com/JungYeoni/da-template/commit/a35916faf97b0f3bcee0e89a0e78625420ab95fa))
- 리포트에 최종 채택 결과 반영 (#17) ([56cf84c](https://github.com/JungYeoni/da-template/commit/56cf84c413eb8ccac018df54957a829a6fed2bf2))
- DATA_CATALOG_raw.md에 checkout product_id 복원 방법 반영 (#20) (#17) ([0a5f73d](https://github.com/JungYeoni/da-template/commit/0a5f73d38973c08ef6ddae384df35f107bc62e68))
- 데이터카탈로그 업데이트 (#17) ([0ff4a84](https://github.com/JungYeoni/da-template/commit/0ff4a8489f41cb42d24362a8804bf0fedddb25c1))
- 프롬프트 카탈로그 문서 업데이트 ([3211722](https://github.com/JungYeoni/da-template/commit/32117221e989ef1807378b80f9a4b6ed9e321d21))
- CHANGELOG 자동 업데이트 [skip ci] ([8896349](https://github.com/JungYeoni/da-template/commit/8896349f97fc083c38585c0d0d34ab823a9c2069))
- Segment cluster k 비교 결과 기록 (#16) ([a644855](https://github.com/JungYeoni/da-template/commit/a6448553b3a3443825dc71ef81cd5f3481b6a1e5))
- CHANGELOG 자동 업데이트 [skip ci] ([919b9b3](https://github.com/JungYeoni/da-template/commit/919b9b373c89cc2952baadd3958c0fd895073df2))
- 페르소나 라벨링 시 LLM 입력 방식 추가 (#15) ([6b3580d](https://github.com/JungYeoni/da-template/commit/6b3580d9f2b0c1ab48ea16e39168784ec0d1a11f))
- 파생 피처 EDA 후 report 업데이트 (#15) ([2d943a9](https://github.com/JungYeoni/da-template/commit/2d943a9f0e9fe70ecabd83f38ff72e4337b1c753))
- CHANGELOG 자동 업데이트 [skip ci] ([5fdfeda](https://github.com/JungYeoni/da-template/commit/5fdfeda01f46d54e9abdf275e0e02875eab05eb7))
- CHANGELOG 자동 업데이트 [skip ci] ([90b0b76](https://github.com/JungYeoni/da-template/commit/90b0b76860e435cc92059432d5ef7a9307d88616))
- Full vs US 페르소나 생성 EDA 추가 #4 ([c23fe47](https://github.com/JungYeoni/da-template/commit/c23fe47d0f6c5eb9df8196484707dcbf0366c170))
- CHANGELOG 자동 업데이트 [skip ci] ([dbadaa8](https://github.com/JungYeoni/da-template/commit/dbadaa833b63c519198ffaca2d39251c851b2737))
- 작업 레포트 및 구현 단계 문서 추가 (#8) ([317e382](https://github.com/JungYeoni/da-template/commit/317e382e972a93ce6e1e340e844bb2dd6ae40794))
- 프롬프트 문서 추가 (#8) ([6b9f967](https://github.com/JungYeoni/da-template/commit/6b9f967fd7114bf7243c48117204f3e2b44258cd))
- CHANGELOG 자동 업데이트 [skip ci] ([fc17a29](https://github.com/JungYeoni/da-template/commit/fc17a299d2b1f9b591694b9580065d95657b6cf5))
- CHANGELOG 자동 업데이트 [skip ci] ([65e072e](https://github.com/JungYeoni/da-template/commit/65e072e9eb9d9316ef876b5a54b85ed874f0073c))
- Report 잘못된 문장 수정 (#5) ([28eac1f](https://github.com/JungYeoni/da-template/commit/28eac1f986160e36fac9d79810389bf2ac8618f1))
- US 고객 분석은 반드시 customers 테이블 기준 US 임을 명시 (#5) ([3c68807](https://github.com/JungYeoni/da-template/commit/3c688078d95dee61006aa923adb750c542eabf73))
- 데이터 카탈로그 문서 추가 (#5) ([a1cebad](https://github.com/JungYeoni/da-template/commit/a1cebad0ff7525032df5225805502dc3b55c7fa7))
- CHANGELOG 자동 업데이트 [skip ci] ([4946bfb](https://github.com/JungYeoni/da-template/commit/4946bfb59419bca9da7a28aa00445c19ef2fdf8d))
- CHANGELOG 자동 업데이트 [skip ci] ([f09c87d](https://github.com/JungYeoni/da-template/commit/f09c87d75bccf9520efae3f4eab94b5114b1a437))
- CHANGELOG 자동 업데이트 [skip ci] ([692e949](https://github.com/JungYeoni/da-template/commit/692e949ede553c3699b865477c56034b5a5ba404))
- CHANGELOG 자동 업데이트 [skip ci] ([c63e500](https://github.com/JungYeoni/da-template/commit/c63e50024621979e023d3be37832e9fdf3d98142))
- GPLR 페르소나 생성 파이프라인 리포트 추가 및 MBA.py 복원 #1 ([be367c4](https://github.com/JungYeoni/da-template/commit/be367c4c9fb4b5dd8a280226ad5b51eb2336189f))
- CHANGELOG 자동 업데이트 [skip ci] ([237313d](https://github.com/JungYeoni/da-template/commit/237313d13868268a17a4d71904da8f309ebe5639))
- CHANGELOG 자동 업데이트 [skip ci] ([e513fa5](https://github.com/JungYeoni/da-template/commit/e513fa5c39eb5c948d8120698a2e6a8e4012a08f))
- CHANGELOG 자동 업데이트 [skip ci] ([d7041cf](https://github.com/JungYeoni/da-template/commit/d7041cff55ad2aad73f8147e3fb7b4cd82591d0b))
- @gustj1819 EDA 결과 문서화 ([0311790](https://github.com/JungYeoni/da-template/commit/0311790de22b217fd0e263b907d2388e8dadbb46))
- CHANGELOG 자동 업데이트 [skip ci] ([b396809](https://github.com/JungYeoni/da-template/commit/b3968098bb1e6b91b63a3a01c8b0fe8797903f02))
- CHANGELOG 자동 업데이트 [skip ci] ([5345095](https://github.com/JungYeoni/da-template/commit/53450958384698186e0b1e1ca44a72a2143e434d))

### 버그 수정

- Assign_segments.py의 segment_common import 경로 수정 (#31) ([0afd7ca](https://github.com/JungYeoni/da-template/commit/0afd7cad6d811fd3a5bfabdf1ce71523e3722593))
- Segment_naming.py의 llm_connector import 경로 수정 (#26) ([8649c67](https://github.com/JungYeoni/da-template/commit/8649c6753df078cc82b0debb2076bb74d09f249c))
- #7 github action CI 테스트 오류 수정 ([a3838ad](https://github.com/JungYeoni/da-template/commit/a3838adc55b112df7e69d6e27195c59dc117f5a4))
- Reference에 recency 컬럼 없을 때 명시적 ValueError 반환 (#16) ([e5cf0d7](https://github.com/JungYeoni/da-template/commit/e5cf0d7a42ae79684241caa63093dec749bbe5af))
- 피드백 반영 후 레포트 업데이트 (#15) ([a847d79](https://github.com/JungYeoni/da-template/commit/a847d7961955b5cec8d0ddd64d1748078a415545))
- CodeRabbit 리뷰 반영 #4 ([1773f1b](https://github.com/JungYeoni/da-template/commit/1773f1bcaf33940cbcb85d0cd510a8172a65f1db))
- LLM 실행 의존성 추가 #4 ([6fef7a2](https://github.com/JungYeoni/da-template/commit/6fef7a29111d42adce2ec6f7fb7e80652f22e4f6))
- LLM 실행 시 dotenv 로드 #4 ([302bf29](https://github.com/JungYeoni/da-template/commit/302bf29ed3aa2139da640307d5b81995de8d4166))
- US-only 페르소나 비교 설정 보정 #4 ([ed597d0](https://github.com/JungYeoni/da-template/commit/ed597d00f6e4826c7d4d2a6b6f71ea373131f819))
- 코드 펜스에 언어 추가 (#8) ([6189d1d](https://github.com/JungYeoni/da-template/commit/6189d1d0e4e09ed64516a38bf98b56334b0ce26a))
- 코드 팬스에 언어 추가 (#8) ([cf360e0](https://github.com/JungYeoni/da-template/commit/cf360e00b971149e3d5da6e59452283c0fee3be3))
- 동일 상품 여러 행이 있으면 마지막 Qunatity만 남기는 경우 수정 (#8) ([19650ac](https://github.com/JungYeoni/da-template/commit/19650ac4fcffbfa14a6eb0ad864611749702aac1))
- Yaml 파일이 없을 시 seed가 달라질 수 있는 문제 방지 (#8) ([527f080](https://github.com/JungYeoni/da-template/commit/527f080ff6a7f079cd876d27796fa2693abf9f36))
- LLM이 응답을 잘못 뱉더라도 아이템 key만 조회하도록 수정 (#8) ([b742747](https://github.com/JungYeoni/da-template/commit/b742747160efc4cc08239a90148de86ec2db6b3e))
- 분할 기준 주석으로 명시 (#8) ([a9cd0be](https://github.com/JungYeoni/da-template/commit/a9cd0bed24d79f7e93779afbaead41d01657174f))
- 각 thread가 독립된 RNG 인스턴스를 가지도록 수정 (#8) ([7459a42](https://github.com/JungYeoni/da-template/commit/7459a42dc629698c4dbc41a0dc2d8bb3eac7fe5d))
- 임포트 정렬 수정 및 CI 오류 수정 (#8) ([eabfe62](https://github.com/JungYeoni/da-template/commit/eabfe62534b06cf3af801b73b8e367e697db643b))
- 데이터 경로 data/raw → data/interim 기준으로 통일 #1 ([6761014](https://github.com/JungYeoni/da-template/commit/6761014926bae8c1fa8561d44b30a198637b95be))
- Ruff 린트 에러 수정 및 논문 구현 코드 lint 예외 처리 #1 ([a439fab](https://github.com/JungYeoni/da-template/commit/a439fab5fdd280e0b7364b16ae5d144a61071074))

### 새 기능

- Feat/20260705_#30_LightGCN_모델_실행_환경_구축_및_평가 (#38)

* feat: LightGCN용 tensorflow 의존성 추가 (#30)

pyproject.toml에 tensorflow>=2.16 추가, uv.lock 동기화.
설치 확인: tensorflow==2.21.0, dense2sparse.py/read_data.py 임포트 정상,
기존 테스트 30/30 통과.

* docs: LightGCN_tri 모델 클래스 설계 문서 작성 (#30)

model_LightGCN_tri 클래스가 실제로는 존재하지 않는다는 것을 발견 —
train_model.py가 참조만 하고 정의가 없음. 신규 구현(A안) vs 기존
model_LGCN_tri 사전학습 경로(B안)를 비교해 A안으로 결정한 근거와
model_LGCN_tri.py 대비 재사용/변경 범위를 기록.

* feat: model_LightGCN_tri 신규 구현 — 표준 LightGCN 전파 (#30)

model_LGCN_tri.py는 사전학습 frequency embedding이 필요한 spectral 방식이라
바로 못 쓴다는 걸 확인하고(docs/LIGHTGCN_TRI_MODEL_DESIGN.md), 정규화
인접행렬을 그대로 곱하는 표준 LightGCN 방식으로 신규 구현했다.

- TDD로 진행: 그래프 빌드/순전파, n_personas=0(#35 bipartite 대비),
  BPR loss, 학습 스텝, 잘못된 optimizer 예외 케이스 5건
- tests/conftest.py 추가: pandas를 먼저 import한 뒤 tensorflow를 import하면
  이 macOS 환경에서 import 자체가 멈추는 문제를 발견해, tensorflow를
  다른 테스트보다 먼저 import하도록 고정해 해결

* fix: LightGCN_tri 실행 경로 legacy 버그 수정 + 실데이터 스모크 테스트 통과 (#30)

model_LightGCN_tri에 keep_prob placeholder 추가(train_model.py/test_model.py
공용 feed_dict 호환용, TDD로 재현 후 수정). 실제 데이터(유저 20,000/상품
1,197/세그먼트 6)로 스모크 테스트하며 아무도 실행해본 적 없던 legacy 버그
4개를 추가로 발견해 수정했다:

- read_data.py: persona_num=20 하드코딩 -> 6 (v2 6-segment)
- read_data.py: all_para 언패킹 리스트 29개 vs 실제 30개 불일치
- train_model 호출 시 all_para[:26]로 잘라 넘겨 17개 요구하는 언패킹 실패
- evaluation.py: `from numpy import *`가 내장 max()를 가려 TypeError

2 epoch 완주, epoch당 약 12.7초 (300 epoch 기준 약 1시간 추정, #34 시간
예산 근거). pyproject.toml에 openpyxl 추가(print_save.py 의존성).

* docs: LightGCN_tri 구현/스모크테스트 결과 리포트 작성 (#30)

* feat: run_lightgcn.py CLI 작성 — 학습+전체유저 추천 저장 (#30)

ALS(als_model.py) 스타일 CLI. model_LightGCN_tri에 top_scores 출력을
추가하고(TDD), train_model()이 F1_max뿐 아니라 sess/model도 반환하도록
확장해 학습 직후 같은 그래프로 전체 유저(샘플 아님) 추천을 뽑을 수 있게
했다. save_recommendations.py를 재사용해 PRED_MAIN_RECOMMEND.csv 저장.

이 파일에서 pandas/tensorflow 임포트 순서 데드락(tests/conftest.py와 같은
원인)이 스크립트 실행 시에도 재현돼, import tensorflow를 최상단으로 옮겨
해결 — ruff의 isort 자동수정이 이 순서를 되돌리려 해서 noqa로 명시 차단.

실행 검증(--epoch 2): 학습 40초 + 전체 유저(20,000명) 추천 생성 10초,
2,000,000행 CSV 정상 생성 확인. 300 epoch 기준 약 1~1.7시간 예상.

configs/LightGCN/params.yaml 신규, data/outputs/LightGCN·logs/LightGCN
gitignore 추가(ALS와 동일 패턴).

* feat: evaluate_lightgcn.py 작성 + 전체 학습 결과 ALS 비교 (#30)

als_evaluate.py와 동일한 구조(사전 계산된 CSV 기반, 모델 재추론 없음)로
HR@K/Recall@K/NDCG@K 평가 스크립트 작성 (TDD, 12건).

300 epoch 전체 학습 실제 실행 결과: 1시간 40분, F1_max=0.03.
평가 결과 ALS(#7) 대비 HR@20 0.0294 vs 0.0608로 낮음 — 하이퍼파라미터
미튜닝 상태의 1회성 비교라 다음 단계(튜닝, event_type 조합 실험)로
재검증 필요. 리포트에 원인 후보와 다음 단계 정리.

* fix: loss 로깅 + 고정 평가셋으로 학습 진단 개선 (#30)

train_model.py: 주석 처리돼 있던 loss 기록을 켜서 epoch마다 저장(Loss 시트).
test_model.py: 매 epoch 무작위 512명 재샘플링 대신(정답 있는 유저가 20,000명
중 1,465명뿐이라 유효 표본이 평균 ~38명으로 노이즈가 컸음) 정답 있는 유저
전체로 고정해 epoch 간 비교가 가능하게 함(fixed_test_batch 옵션 추가,
기존 호출부는 하위 호환).

같은 하이퍼파라미터로 재실행한 결과 HR@20 0.0294 → 0.0403(+37%) — 모델이
아니라 측정 문제였음을 확인. loss는 마지막 20 epoch에서 정체(수렴 완료).

docs/LIGHTGCN_TRI_TUNING_PLAN.md 신규 — 다음 개선 우선순위(event_type 조합
→ 하이퍼파라미터 튜닝)와 bipartite(#34) 비교 조건 통일 표 정리.
리포트에 2차 실행 결과 및 해석 반영.

* feat: make_lgcn_graph.py에 --event-types 옵션 추가 (#30)

u2t event_type 조합 실험용 (TDD, 3건). purchase만 / add_to_cart+purchase /
전부(기본값) 세 가지로 실제 학습·평가한 결과, "구매만"이 HR@20 0.0403 ->
0.0491(+22%)로 가장 좋았고 ALS와의 격차도 1.51배 -> 1.24배로 좁혀짐.
반대로 add_to_cart+purchase 조합은 전부 포함보다도 나빴음(HR@20=0.0389) —
장바구니 신호가 구매 의도와 안 맞는 경우가 많아 오히려 노이즈로 작용한
것으로 추정. 상세 결과는 후속 커밋의 리포트/이슈 코멘트에 정리.

* feat: 3차 튜닝(event_type/하이퍼파라미터/negative sampling) 결과 반영 (#30)

event_type 조합, emb_dim×lr 9개 그리드, lamda 라운드, negative sampling
라운드까지 순차 실험한 결과를 리포트에 통합 정리.

최종 확정: event_type=구매만, emb_dim=32, lr=0.005, lamda=0.02(원래값),
neg_samples=1(기존 방식 유지) — HR@20 0.0294(1차) -> 0.0553, ALS 대비
격차 2.07배 -> 1.10배로 축소.

negative sampling 실험용으로 parse.py/params.py에 --sample_rate 추가,
train_model.py의 LightGCN_tri를 SAMPLE_RATE 적용 모델 목록에 포함
(기존엔 목록에 없어서 항상 negative 1개로 강제됐음). RecBole 제안
reg_weight, SimpleX 제안 negative 다중 샘플링 둘 다 우리 데이터에선
반대 결과 — 원인과 재현성 한계는 리포트에 명시.

* fix: lgcn3 bare 임포트 모듈을 known-first-party에 등록 (CI ruff I001) (#30)

test_model.py의 import 순서가 로컬 pre-commit 훅 실행마다 로컬<->CI
결과가 달라져 왔다갔다 했던 원인 — evaluation 모듈이 known-first-party
목록에 없어 정렬 기준이 모호했던 것. src/baselines/lgcn3/ 내 bare
임포트로 쓰는 형제 모듈 전체를 등록해 로컬/CI 판정을 일치시켰다.

* fix: 나머지 lgcn3 파일들 import 순서 정리 (CI ruff I001) (#30)

known-first-party 등록만으로는 안 잡히던 그룹 간 빈 줄 누락을
ruff --fix로 일괄 정리. 전부 공백/순서 변경만 있고 동작 변화 없음
(pytest 54/54 통과 재확인).

* fix: CodeRabbit 지적 실버그 3건 수정 (#30)

1. layer_weight가 1/(l+1)로 레이어 인덱스에 따라 줄어들어 원본 임베딩에
   편중돼 있었음 — 표준 LightGCN처럼 모든 레이어에 균등한 1/(layer+1)로
   수정. 지금까지의 전체 튜닝 결과가 이 버그 위에서 나온 것이라 재실험 필요.
2. bpr_loss의 log(sigmoid(x))가 sigmoid 포화 시 -inf가 될 수 있어
   log_sigmoid(x)로 교체 (수치안정성).
3. train_model.py의 fixed_test_batch가 모델 종류 무관하게 전체 적용되던
   것을 LightGCN_tri 전용으로 제한 — 다른 레거시 모델은 기존 무작위
   샘플링 동작 유지.

TDD: layer_weight 균등성, bpr_loss 유한값 테스트 추가 (56/56 통과).

* fix: 임베딩 초기화에 재현성 seed 추가 (#30)

tf.random.normal에 seed가 없어 user/item/persona_embeddings 초기값이
실행마다 달라졌음 — CLAUDE.md 재현성 규칙(random_state=42) 위반
(CodeRabbit 지적). tf.compat.v1.set_random_seed(42)로 고정.

TDD: 동일 조건 2회 빌드 시 초기 임베딩이 같은지 검증하는 테스트 추가
(57/57 통과). ([84f1c63](https://github.com/JungYeoni/da-template/commit/84f1c6327c0c77a8cc1659cc02d8ee93f2aac3e1))
- LightGCN bipartite(u2t만) 그래프 데이터 생성 모드 추가 (#35) ([ce45a95](https://github.com/JungYeoni/da-template/commit/ce45a95e5fbe864072d3181e84b2a01d721613bd))
- LightGCN용 tri-graph 데이터 파이프라인 구축 (#29) ([81ec74a](https://github.com/JungYeoni/da-template/commit/81ec74a8cbb29f97513fba17af216b04833ca234))
- LightGCN tri-graph에 item-segment lift 가중치 지원 추가 (#29) ([dec323f](https://github.com/JungYeoni/da-template/commit/dec323f8a283797635bd3bec45dd465654a0fe1a))
- User_id/item_id 인덱스 인코딩 공용 유틸 추가 (#29) ([9ea307f](https://github.com/JungYeoni/da-template/commit/9ea307f364b637595f7138f1c6d88746f305a3e1))
- LightGCN용 train 전용 세그먼트 재계산 파이프라인 추가 (#31) ([01a3052](https://github.com/JungYeoni/da-template/commit/01a3052b03b73ebb15c08d4283986d97c260be59))
- 고객별 세그먼트 라벨 테이블 생성 #26 ([0496817](https://github.com/JungYeoni/da-template/commit/0496817da6a0f74eb749695d3ea4500189507119))
- 세그먼트 병합 시 evidence/cautions/status/erros 보존 (#26) ([8246585](https://github.com/JungYeoni/da-template/commit/8246585b6f228f9c07f1c105e0841a6449779901))
- V1 vs v2 세그먼트 품질 비교 리포트 (#18) ([ddd7173](https://github.com/JungYeoni/da-template/commit/ddd7173b0ce7677046c330afd25298d9805bc775))
- US-only 분석 트랙 제거 — Full 데이터로 통합 #23 ([a0d0c1d](https://github.com/JungYeoni/da-template/commit/a0d0c1df5accf1598f38dadf798d24e02f59f8eb))
- Demographic/lifestyle 금지어 소프트체크 범위 확장 (#17) ([a1bff38](https://github.com/JungYeoni/da-template/commit/a1bff382dffb3ac599713b7004deb2e914f9d342))
- #20 이슈 검증 (#7) ([2560420](https://github.com/JungYeoni/da-template/commit/25604207823195284d8a0da5c2fe1a59476e2fbc))
- Gitignore 경로 수정. 필요한 디렉토리 경로 추가 ([e7868e0](https://github.com/JungYeoni/da-template/commit/e7868e0e73e4a9363e1570275b8c4056b74c0685))
- #7 ALS 학습 및 평가 코드(평가지표: HR@K, NDCG@K, Recall@K) ([cd1de63](https://github.com/JungYeoni/da-template/commit/cd1de63ab0e852a60d7565dab7ada3a702e83701))
- #7 ALS 데이터마트 생성 코드 ([3baffa5](https://github.com/JungYeoni/da-template/commit/3baffa597933d7124d7e07fccbb2c761f5b7cb38))
- Segment naming v2 최종 채택 — run_2026-07-03_1 (#17) ([3cbbf76](https://github.com/JungYeoni/da-template/commit/3cbbf769bf1ff85c33590f3a419421e137a62cf9))
- Naming temperature 고정 및 실험 버전 관리 도입 (#17) ([da88f5c](https://github.com/JungYeoni/da-template/commit/da88f5c5946e98dbd6dc6e99a9741cf7ca9aa39b))
- Segment naming 산출물 추적 및 재생성 (#17) ([ab83ed0](https://github.com/JungYeoni/da-template/commit/ab83ed0dbefc012f78aae1552c1bee170a378e69))
- 페르소나명 라벨링 ([558b453](https://github.com/JungYeoni/da-template/commit/558b453e6597d5976ef7dd4155937c09e841ffb6))
- Clustering 기반 segment assignment 추가 (#16) ([5059633](https://github.com/JungYeoni/da-template/commit/505963328b56f42fbc60b582db8b9f3ba90ec5a1))
- Segment 입력 피처 생성 추가 (#16) ([15cbc63](https://github.com/JungYeoni/da-template/commit/15cbc63f56144c09fb40e7782d4902c9d5db7394))
- 세그먼트를 위한 파생피처 생성 후 EDA (#15) ([e1e8e8c](https://github.com/JungYeoni/da-template/commit/e1e8e8c367d93b6fb9d0531995d475f74787a568))
- 전체 데이터 vs US-only 페르소나 설정 추가 #4 ([5aa617a](https://github.com/JungYeoni/da-template/commit/5aa617ae711b01f363f65e273e633197df0e3b3d))
- Llm conector test 구현 (#8) ([3902d5e](https://github.com/JungYeoni/da-template/commit/3902d5edf5d75ff9040a8cec07ad87d516f940ca))
- 현재 구현된 코드에 맞게 라이브러리 및 경로 수정 (#8) ([9a3387f](https://github.com/JungYeoni/da-template/commit/9a3387f13d54271c767ca68816f0ee87a31dceb6))
- LLM 파이프라인 병렬화 (#8) ([d1397ed](https://github.com/JungYeoni/da-template/commit/d1397edffafe5650a2d6948414734bb57cdebc70))
- 빈 커밋 ([45193b4](https://github.com/JungYeoni/da-template/commit/45193b4de69066b7ea135c6800a5c54fbb704677))
- 실제 저장 파일명에 맞게 수정 (#5) ([736ef22](https://github.com/JungYeoni/da-template/commit/736ef22979c8031cf766a2753c7c3d66b9ee05ee))
- 상대경로 수정 및 고객 수 하드코딩 제거 (#5) ([001e5c9](https://github.com/JungYeoni/da-template/commit/001e5c9bf566296a622c6f74980695ef1a73250f))
- 무결성 검증 예외 전파 추가 (#5) ([2a6657d](https://github.com/JungYeoni/da-template/commit/2a6657df71c5fffcf93e1003511d161b8a3ded99))
- Us 필터링 기준 주석으로 명시 (#5) ([5695e08](https://github.com/JungYeoni/da-template/commit/5695e08e68496083c7ffbaa93359ef2db403f8c2))
- 주문 단위 금액이 중복으로 집계되는 경우 수정 (#5) ([d33258e](https://github.com/JungYeoni/da-template/commit/d33258e7286a01e24e43043bf42a9338f5d49600))
- 고객 단위 집계 파이프라인 및 EDA 보고서 추가 #5 ([efcb1b5](https://github.com/JungYeoni/da-template/commit/efcb1b5f5690fde905d8d1b47e1f21dd040b420e))
- 집계 테이블 eda #5 ([5b389c3](https://github.com/JungYeoni/da-template/commit/5b389c3a77bef10b84b441968f111a362511605e))
- 고객 단위 집계 파이프라인 구축 (전체 / US-only 공용) #5 ([3c4d703](https://github.com/JungYeoni/da-template/commit/3c4d70317f6bd3fba8f0b588ad3dafc6ba44d457))
- 데이터 불러오기 코드 주석 처리 #2 ([3493967](https://github.com/JungYeoni/da-template/commit/349396708b1bda7b156c7f386abae7dc559b14e8))
- @eric010314-sys 작업 내용 transfer #1 ([37fae63](https://github.com/JungYeoni/da-template/commit/37fae630bbc760e447f35c1d7f6f3d9510d4930a))

### 테스트

- Tri-graph 매핑 로직 단위 테스트 추가 (#29) ([46193fb](https://github.com/JungYeoni/da-template/commit/46193fb09481355b27d6e8c9ac7e6bc8f7ad02b8))
- 세그먼트 병합/검증 로직 단위 테스트 추가 (#26) ([35114fd](https://github.com/JungYeoni/da-template/commit/35114fd1981d2a6202d39763487aab667115ff84))


