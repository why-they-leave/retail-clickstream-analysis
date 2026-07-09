"""save_recommendations.py 단위 테스트 (Issue #34 후속 — bipartite/tri 파일명 충돌 수정).

graph_mode별로 다른 CSV 파일에 저장해야 tri/bipartite 실행이 서로 덮어쓰지 않는다.
rec-system 레포(backend/api/services/lightgcn_service.py)가 쓰는 graph_type
어휘("bipartite"/"tripartite")에 맞춰 파일명을 짓는다.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "baselines" / "lgcn3"))

from save_recommendations import (  # noqa: E402
    REQUIRED_COLUMNS,
    resolve_rec_filename,
    save_lightgcn_recommendations,
)


class TestResolveRecFilename:
    def test_tri_mode_maps_to_tripartite_filename(self):
        assert resolve_rec_filename("tri") == "PRED_MAIN_RECOMMEND_tripartite.csv"

    def test_bipartite_mode_maps_to_bipartite_filename(self):
        assert resolve_rec_filename("bipartite") == "PRED_MAIN_RECOMMEND_bipartite.csv"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_rec_filename("unknown")


class TestSaveLightgcnRecommendations:
    def _make_df(self):
        return pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": ["P1", "P2"],
                "score": [0.9, 0.5],
                "rank": [1, 1],
            }
        )

    def test_tri_and_bipartite_write_to_different_files(self, tmp_path):
        df = self._make_df()

        tri_path = save_lightgcn_recommendations(df, graph_mode="tri", output_dir=tmp_path)
        bipartite_path = save_lightgcn_recommendations(
            df, graph_mode="bipartite", output_dir=tmp_path
        )

        assert tri_path != bipartite_path
        assert tri_path.exists()
        assert bipartite_path.exists()

    def test_missing_required_column_raises(self, tmp_path):
        df = self._make_df().drop(columns=["score"])
        with pytest.raises(ValueError):
            save_lightgcn_recommendations(df, graph_mode="tri", output_dir=tmp_path)

    def test_saved_csv_has_model_type_column(self, tmp_path):
        df = self._make_df()
        path = save_lightgcn_recommendations(df, graph_mode="tri", output_dir=tmp_path)
        saved = pd.read_csv(path)
        assert set(REQUIRED_COLUMNS + ["model_type"]).issubset(saved.columns)
