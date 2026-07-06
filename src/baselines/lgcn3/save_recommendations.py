"""LightGCN 추천 결과를 PRED_MAIN_RECOMMEND.csv로 저장한다 (Issue #29).

rec-system 레포 Issue #4가 정의한 스키마(user_id, item_id, score, rank, model_type)를
따른다. #30(모델 학습/평가)에서 추천 결과 DataFrame을 만든 뒤 이 함수를 호출하면 된다.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "LightGCN"
REC_FILE = "PRED_MAIN_RECOMMEND.csv"

REQUIRED_COLUMNS = ["user_id", "item_id", "score", "rank"]


def save_lightgcn_recommendations(df_rec: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    """추천 결과 DataFrame에 model_type="LightGCN"을 붙여 저장한다.

    df_rec는 최소 user_id/item_id/score/rank 컬럼을 가져야 한다(rec-system #4 스키마).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df_rec.columns]
    if missing:
        raise ValueError(f"df_rec에 필요한 컬럼이 없습니다: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = df_rec.copy()
    result["model_type"] = "LightGCN"

    output_path = output_dir / REC_FILE
    result.to_csv(output_path, index=False)
    logger.info("[저장] %s (%s개 레코드)", output_path, len(result))
    return output_path
