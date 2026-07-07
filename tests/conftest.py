"""pandas보다 tensorflow를 먼저 임포트해 macOS 임포트 순서 데드락을 피한다.

이 프로젝트 환경(macOS, Accelerate/BLAS)에서 pandas를 먼저 임포트한 뒤
tensorflow를 임포트하면 tensorflow의 `import` 자체가 데드락에 빠지는 문제가
확인됐다(tests/test_model_lightgcn_tri.py 작업 중 발견, TF/pandas 단독 임포트는
각각 정상 동작 — 두 라이브러리를 같은 프로세스에서 이 순서로 쓸 때만 발생).

pytest는 각 디렉터리의 conftest.py를 다른 테스트 모듈보다 먼저 로드하므로,
여기서 tensorflow를 가장 먼저 임포트해두면 이후 어떤 순서로 테스트 파일이
수집되더라도(예: pandas를 쓰는 test_features.py가 먼저 수집돼도) 안전하다.
"""

import tensorflow  # noqa: F401
