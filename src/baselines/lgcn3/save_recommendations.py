"""LightGCN 추천 결과를 PRED_MAIN_RECOMMEND_<graph_mode>.csv로 저장한다 (Issue #29, #34).

rec-system 레포 Issue #4가 정의한 스키마(user_id, item_id, score, rank, model_type)를
따른다. #30(모델 학습/평가)에서 추천 결과 DataFrame을 만든 뒤 이 함수를 호출하면 된다.

graph_mode(tri/bipartite)별로 파일명을 분리한다 — 예전에는 고정 파일명
(PRED_MAIN_RECOMMEND.csv)을 써서 tri/bipartite를 번갈아 실행하면 서로 덮어썼다
(#34 bipartite 대조군 실험 도입 후 발견). rec-system 쪽 graph_type 어휘
("bipartite"/"tripartite", backend/api/services/lightgcn_service.py 참고)에
맞춰 파일명을 지어서, 추후 rec-system이 두 파일을 각각 읽도록 연결하기 쉽게 한다.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN"

REQUIRED_COLUMNS = ["user_id", "item_id", "score", "rank"]

GRAPH_MODE_FILE_SUFFIX = {"tri": "tripartite", "bipartite": "bipartite"}


def resolve_rec_filename(graph_mode: str) -> str:
    """graph_mode(tri/bipartite)에 맞는 CSV 파일명을 반환한다."""
    if graph_mode not in GRAPH_MODE_FILE_SUFFIX:
        raise ValueError(f"알 수 없는 graph_mode: {graph_mode}")
    return f"PRED_MAIN_RECOMMEND_{GRAPH_MODE_FILE_SUFFIX[graph_mode]}.csv"


def save_lightgcn_recommendations(
    df_rec: pd.DataFrame, graph_mode: str, output_dir: Path = OUTPUT_DIR
) -> Path:
    """추천 결과 DataFrame에 model_type="LightGCN"을 붙여 graph_mode별 파일로 저장한다.

    df_rec는 최소 user_id/item_id/score/rank 컬럼을 가져야 한다(rec-system #4 스키마).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df_rec.columns]
    if missing:
        raise ValueError(f"df_rec에 필요한 컬럼이 없습니다: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = df_rec.copy()
    result["model_type"] = "LightGCN"

    output_path = output_dir / resolve_rec_filename(graph_mode)
    result.to_csv(output_path, index=False)
    logger.info("[저장] %s (%s개 레코드)", output_path, len(result))
    return output_path
