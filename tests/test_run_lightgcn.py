"""run_lightgcn.py의 순수 함수 단위 테스트 (Issue #30).

TF 세션이 필요한 학습/추론 부분은 스모크 테스트(reports/[ML]_LightGCN_tri_...)로
검증했고, 여기서는 그 출력을 CSV 스키마로 변환하는 순수 로직만 검증한다.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "baselines" / "lgcn3"))

from run_lightgcn import build_recommendation_df  # noqa: E402


class TestBuildRecommendationDf:
    def test_columns_match_save_recommendations_schema(self):
        """save_recommendations.py의 REQUIRED_COLUMNS(user_id/item_id/score/rank)를 만족해야 함."""
        top_items = np.array([[10, 11], [12, 13]])
        top_scores = np.array([[0.9, 0.5], [0.8, 0.3]])
        user_ids = [0, 1]
        item_dec = {10: "P010", 11: "P011", 12: "P012", 13: "P013"}

        df = build_recommendation_df(top_items, top_scores, user_ids, item_dec)

        assert set(["user_id", "item_id", "score", "rank"]).issubset(df.columns)

    def test_item_indices_decoded_to_original_ids(self):
        top_items = np.array([[10, 11]])
        top_scores = np.array([[0.9, 0.5]])
        user_ids = [0]
        item_dec = {10: "P010", 11: "P011"}

        df = build_recommendation_df(top_items, top_scores, user_ids, item_dec)

        assert list(df["item_id"]) == ["P010", "P011"]

    def test_rank_is_1_indexed_and_matches_score_order(self):
        """top_k(sorted=True) 출력이므로 순서 그대로 1,2,3...을 rank로 매긴다."""
        top_items = np.array([[10, 11, 12]])
        top_scores = np.array([[0.9, 0.5, 0.1]])
        user_ids = [7]
        item_dec = {10: "P010", 11: "P011", 12: "P012"}

        df = build_recommendation_df(top_items, top_scores, user_ids, item_dec)

        assert list(df["rank"]) == [1, 2, 3]
        assert (df["user_id"] == 7).all()

    def test_multiple_users_produce_independent_rank_sequences(self):
        top_items = np.array([[10, 11], [12, 13]])
        top_scores = np.array([[0.9, 0.5], [0.8, 0.3]])
        user_ids = [0, 5]
        item_dec = {10: "P010", 11: "P011", 12: "P012", 13: "P013"}

        df = build_recommendation_df(top_items, top_scores, user_ids, item_dec)

        assert len(df) == 4
        assert list(df[df["user_id"] == 0]["rank"]) == [1, 2]
        assert list(df[df["user_id"] == 5]["rank"]) == [1, 2]
