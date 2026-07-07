"""LightGCN용 그래프 매핑 로직 단위 테스트 (tri: Issue #29, bipartite: Issue #35)."""

import pandas as pd
import pytest

from src.datasets.make_lgcn_graph import (
    LIFT_THRESHOLD,
    MIN_PURCHASE_COUNT,
    build_empty_mapping,
    build_t2p_mapping,
    build_u2p_mapping,
    build_u2t_mapping,
)

# ── 픽스처 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def user_enc() -> dict:
    """customer_id 101~103 -> uidx 0~2."""
    return {101: 0, 102: 1, 103: 2}


@pytest.fixture
def item_enc() -> dict:
    """product_id 201~203 -> tidx 0~2."""
    return {201: 0, 202: 1, 203: 2}


# ── build_u2t_mapping ────────────────────────────────────────────────────────


class TestBuildU2tMapping:
    def test_all_users_present_even_without_interaction(self, user_enc, item_enc):
        pairs = pd.DataFrame({"customer_id": [101], "product_id": [201]})
        result = build_u2t_mapping(pairs, user_enc, item_enc)
        assert set(result.keys()) == {0, 1, 2}
        assert result[0] == [0]
        assert result[1] == []
        assert result[2] == []

    def test_duplicate_pairs_are_deduplicated(self, user_enc, item_enc):
        pairs = pd.DataFrame({"customer_id": [101, 101, 101], "product_id": [201, 201, 202]})
        result = build_u2t_mapping(pairs, user_enc, item_enc)
        assert result[0] == [0, 1]

    def test_unknown_ids_are_dropped(self, user_enc, item_enc):
        pairs = pd.DataFrame({"customer_id": [999], "product_id": [201]})
        result = build_u2t_mapping(pairs, user_enc, item_enc)
        assert all(v == [] for v in result.values())


# ── build_u2p_mapping ────────────────────────────────────────────────────────


class TestBuildU2pMapping:
    def test_single_label_per_user(self, user_enc):
        segment_labeled = pd.DataFrame({"customer_id": [101, 102, 103], "segment_id": [0, 3, 0]})
        result = build_u2p_mapping(user_enc, segment_labeled)
        assert result == {0: [0], 1: [3], 2: [0]}


# ── build_t2p_mapping ────────────────────────────────────────────────────────


@pytest.fixture
def segment_labeled_balanced() -> pd.DataFrame:
    """30명 고객, segment 0/1/2에 10명씩 균등 배정 (segment_share = 1/3)."""
    customer_ids = list(range(1, 31))
    segment_ids = [0] * 10 + [1] * 10 + [2] * 10
    return pd.DataFrame({"customer_id": customer_ids, "segment_id": segment_ids})


class TestBuildT2pMapping:
    def test_single_segment_connection_when_only_one_segment_buys(self, segment_labeled_balanced):
        """상품 A: segment 0 고객 5명만 구매 -> segment 0에만 연결(lift=3.0)."""
        item_enc = {"A": 0}
        orders = pd.DataFrame({"customer_id": list(range(1, 6)), "product_id": ["A"] * 5})
        result = build_t2p_mapping(orders, item_enc, segment_labeled_balanced)

        assert len(result[0]) == 1
        segment_id, lift = result[0][0]
        assert segment_id == 0
        assert lift == pytest.approx(3.0)

    def test_multi_segment_connection_when_multiple_segments_pass_threshold(
        self, segment_labeled_balanced
    ):
        """상품 B: segment 0 3명 + segment 1 3명 구매 -> 둘 다 연결(다중 연결, 단일 아님)."""
        item_enc = {"B": 0}
        orders = pd.DataFrame({"customer_id": [1, 2, 3, 11, 12, 13], "product_id": ["B"] * 6})
        result = build_t2p_mapping(orders, item_enc, segment_labeled_balanced)

        connected_segments = {segment_id for segment_id, _ in result[0]}
        assert connected_segments == {0, 1}
        assert len(result[0]) == 2

    def test_below_min_purchase_count_is_excluded(self, segment_labeled_balanced):
        """구매 건수가 MIN_PURCHASE_COUNT 미만이면 lift 불안정으로 제외."""
        assert MIN_PURCHASE_COUNT >= 4
        item_enc = {"C": 0}
        orders = pd.DataFrame({"customer_id": [1, 2, 3], "product_id": ["C"] * 3})
        result = build_t2p_mapping(orders, item_enc, segment_labeled_balanced)
        assert result[0] == []

    def test_all_items_present_even_without_connection(self, segment_labeled_balanced):
        """item_num은 t2p 딕셔너리 길이로 결정되므로 전체 상품이 키로 존재해야 함."""
        item_enc = {"A": 0, "B": 1, "C": 2}
        orders = pd.DataFrame({"customer_id": list(range(1, 6)), "product_id": ["A"] * 5})
        result = build_t2p_mapping(orders, item_enc, segment_labeled_balanced)
        assert set(result.keys()) == {0, 1, 2}
        assert result[1] == []
        assert result[2] == []

    def test_below_threshold_lift_does_not_connect(self, segment_labeled_balanced):
        """모든 segment에서 고르게 팔리면(lift ~= 1.0) threshold 미만이라 연결 안 됨."""
        item_enc = {"D": 0}
        # segment 0/1/2에서 각 2명씩 구매 -> 각 segment lift = (2/6)/(1/3) = 1.0
        orders = pd.DataFrame({"customer_id": [1, 2, 11, 12, 21, 22], "product_id": ["D"] * 6})
        result = build_t2p_mapping(orders, item_enc, segment_labeled_balanced)
        assert LIFT_THRESHOLD > 1.0
        assert result[0] == []


# ── build_empty_mapping (Issue #35 — bipartite 모드의 u2p/t2p) ─────────────────


class TestBuildEmptyMapping:
    def test_all_keys_present_with_empty_lists(self):
        """bipartite 모드에서는 u2p/t2p를 계산하지 않고, 키만 전부 채운 빈 매핑을 만든다."""
        result = build_empty_mapping(3)
        assert result == {0: [], 1: [], 2: []}

    def test_zero_count_returns_empty_dict(self):
        result = build_empty_mapping(0)
        assert result == {}
