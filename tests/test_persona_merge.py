"""고객별 세그먼트 라벨 병합 단위 테스트 (Issue #26)."""

import json

import pandas as pd
import pytest

from src.persona.segment_naming import merge_personas_with_customers, validate_merged_segments

# ── 픽스처 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def customer_segments() -> pd.DataFrame:
    """segment_id 0, 1에 배정된 고객 4명."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "segment_id": [0, 0, 1, 1],
        }
    )


@pytest.fixture
def personas_success() -> list[dict]:
    """두 segment 모두 naming이 성공한 케이스."""
    return [
        {
            "segment_id": 0,
            "segment_name": "Segment A",
            "description": "desc A",
            "evidence": ["ev1", "ev2"],
            "cautions": ["caution1"],
        },
        {
            "segment_id": 1,
            "segment_name": "Segment B",
            "description": "desc B",
            "evidence": ["ev3"],
            "cautions": [],
        },
    ]


# ── merge_personas_with_customers ──────────────────────────────────────────────


class TestMergePersonasWithCustomers:
    def test_merges_segment_name_and_description(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        assert merged.loc[merged["segment_id"] == 0, "segment_name"].unique().tolist() == [
            "Segment A"
        ]
        assert merged.loc[merged["segment_id"] == 1, "description"].unique().tolist() == ["desc B"]

    def test_evidence_and_cautions_are_json_strings(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        evidence_value = merged.loc[merged["segment_id"] == 0, "evidence"].iloc[0]
        assert json.loads(evidence_value) == ["ev1", "ev2"]

    def test_row_count_preserved(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        assert len(merged) == len(customer_segments)

    def test_failed_naming_segment_keeps_status_and_errors(self, customer_segments):
        personas = [
            {
                "segment_id": 0,
                "segment_name": "Segment A",
                "description": "desc A",
                "evidence": ["ev1"],
                "cautions": [],
            },
            {"segment_id": 1, "status": "NAMING_FAILED", "errors": ["필수 키 누락"]},
        ]
        merged = merge_personas_with_customers(customer_segments, personas)
        failed_rows = merged[merged["segment_id"] == 1]
        assert failed_rows["segment_name"].isna().all()
        assert (failed_rows["status"] == "NAMING_FAILED").all()

    def test_all_naming_failed_does_not_raise_keyerror(self, customer_segments):
        """모든 segment naming이 실패하면 evidence/cautions 등 컬럼이 아예 안 생길 수 있다."""
        personas = [
            {"segment_id": 0, "status": "NAMING_FAILED", "errors": ["e1"]},
            {"segment_id": 1, "status": "NAMING_FAILED", "errors": ["e2"]},
        ]
        merged = merge_personas_with_customers(customer_segments, personas)
        assert merged["segment_name"].isna().all()
        assert len(merged) == len(customer_segments)


# ── validate_merged_segments ────────────────────────────────────────────────────


class TestValidateMergedSegments:
    def test_passes_on_valid_merge(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        validate_merged_segments(merged, customer_segments, personas_success)

    def test_raises_on_missing_segment_name(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        merged.loc[0, "segment_name"] = pd.NA
        with pytest.raises(ValueError, match="segment_name"):
            validate_merged_segments(merged, customer_segments, personas_success)

    def test_raises_on_row_count_mismatch(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        merged = pd.concat([merged, merged.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="row 수 불일치"):
            validate_merged_segments(merged, customer_segments, personas_success)

    def test_raises_on_segment_count_mismatch(self, customer_segments, personas_success):
        merged = merge_personas_with_customers(customer_segments, personas_success)
        personas_extra = [
            *personas_success,
            {
                "segment_id": 2,
                "segment_name": "Segment C",
                "description": "desc C",
                "evidence": [],
                "cautions": [],
            },
        ]
        with pytest.raises(ValueError, match="segment 수 불일치"):
            validate_merged_segments(merged, customer_segments, personas_extra)
