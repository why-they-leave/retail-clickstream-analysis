"""evaluate_lightgcn.py 단위 테스트 (Issue #30). als_evaluate.py와 동일한 지표 로직."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "baselines" / "lgcn3"))

from evaluate_lightgcn import (  # noqa: E402
    build_ground_truth,
    evaluate_at_k,
    hit_rate_at_k,
    load_valid_pairs_as_df,
    ndcg_at_k,
    recall_at_k,
)


class TestHitRateAtK:
    def test_hit_when_overlap_exists(self):
        assert hit_rate_at_k([1, 2, 3], {2, 5}) == 1.0

    def test_no_hit_when_no_overlap(self):
        assert hit_rate_at_k([1, 2, 3], {5, 6}) == 0.0


class TestRecallAtK:
    def test_partial_recall(self):
        assert recall_at_k([1, 2], {1, 2, 3, 4}) == pytest.approx(0.5)

    def test_empty_ground_truth_returns_zero(self):
        assert recall_at_k([1, 2], set()) == 0.0


class TestNdcgAtK:
    def test_perfect_rank_gives_ndcg_1(self):
        assert ndcg_at_k([1, 2], {1, 2}) == pytest.approx(1.0)

    def test_no_hit_gives_ndcg_0(self):
        assert ndcg_at_k([9, 8], {1, 2}) == 0.0

    def test_worse_rank_scores_lower_than_perfect_rank(self):
        perfect = ndcg_at_k([1, 2], {1, 2})
        worse = ndcg_at_k([2, 1], {1})  # 정답이 1개인데 뒤쪽에 옴
        assert worse < perfect


class TestBuildGroundTruth:
    def test_groups_items_per_user(self):
        test_df = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 11, 20]})
        gt = build_ground_truth(test_df)
        assert gt == {1: {10, 11}, 2: {20}}


class TestEvaluateAtK:
    def test_computes_hr_recall_ndcg_for_k(self):
        recs_df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": [10, 99, 98],
                "rank": [1, 2, 3],
            }
        )
        ground_truth = {1: {10}}
        result = evaluate_at_k(recs_df, ground_truth, k=1)
        assert result["k"] == 1
        assert result["HR"] == 1.0
        assert result["eval_users"] == 1

    def test_skips_users_with_no_recommendations(self):
        recs_df = pd.DataFrame({"user_id": [1], "item_id": [10], "rank": [1]})
        ground_truth = {1: {10}, 2: {20}}  # 유저 2는 추천 결과 없음
        result = evaluate_at_k(recs_df, ground_truth, k=1)
        assert result["eval_users"] == 1
        assert result["skipped"] == 1


class TestLoadValidPairsAsDf:
    def test_decodes_uidx_tidx_to_original_ids(self):
        valid_data = {"0": [5], "1": []}
        user_dec = {0: 101, 1: 102}
        item_dec = {5: "P005"}

        df = load_valid_pairs_as_df(valid_data, user_dec, item_dec)

        assert list(df.columns) == ["user_id", "item_id"]
        assert len(df) == 1
        assert df.iloc[0]["user_id"] == 101
        assert df.iloc[0]["item_id"] == "P005"

    def test_users_with_no_valid_items_produce_no_rows(self):
        valid_data = {"0": []}
        user_dec = {0: 101}
        item_dec = {}

        df = load_valid_pairs_as_df(valid_data, user_dec, item_dec)

        assert len(df) == 0
