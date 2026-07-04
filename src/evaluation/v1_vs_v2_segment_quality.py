"""
v1(LLM 직접 생성 persona) vs v2(KMeans + LLM naming) 세그먼트 품질 비교 지표 (Issue #18).

Full 데이터 단일 트랙 기준(Issue #23 반영) — Full vs US 비교는 하지 않는다.

입력:
    data/interim/funnel_persona_gen/user_persona_labels_v1.json  (v1, 유저 라벨링만)
    data/processed/customer_segments_all_customers.csv            (v2, #16 산출물)
    data/processed/segment_personas_v2.json                       (v2, #17 채택본)

출력:
    data/processed/v1_vs_v2_segment_quality.csv
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INTERIM_DIR = ROOT_DIR / "data" / "interim"

V1_LABELS_PATH = INTERIM_DIR / "funnel_persona_gen" / "user_persona_labels_v1.json"
V2_CUSTOMER_SEGMENTS_PATH = PROCESSED_DIR / "customer_segments_all_customers.csv"
V2_PERSONAS_PATH = PROCESSED_DIR / "segment_personas_v2.json"
OUTPUT_PATH = PROCESSED_DIR / "v1_vs_v2_segment_quality.csv"


def shannon_entropy(ratios: pd.Series) -> float:
    """비율 분포의 Shannon entropy(bit 단위)를 계산한다. 0 비율은 제외."""
    nonzero = ratios[ratios > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def normalized_entropy(ratios: pd.Series) -> float:
    """entropy를 log2(그룹 수)로 정규화해 0~1 범위로 만든다."""
    n = len(ratios)
    if n <= 1:
        return np.nan
    return shannon_entropy(ratios) / np.log2(n)


def top_k_concentration(ratios: pd.Series, k: int) -> float:
    """상위 k개 그룹의 비율 합을 계산한다."""
    return float(ratios.sort_values(ascending=False).head(k).sum())


def compute_v1_metrics() -> dict:
    """v1 유저 라벨링 결과(다중 라벨)의 분포 품질 지표를 계산한다."""
    with open(V1_LABELS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    sample_size = data["sample_size"]
    total_users = data["total_users"]
    labels = data["labels"]

    ok_labels = [label for label in labels if label["status"] == "ok"]
    all_persona_picks = [persona for label in ok_labels for persona in label["personas"]]
    persona_counts = Counter(all_persona_picks)

    total_picks = sum(persona_counts.values())
    ratios = pd.Series({p: c / total_picks for p, c in persona_counts.items()})

    avg_personas_per_labeled_user = total_picks / len(ok_labels) if ok_labels else np.nan

    return {
        "method": "v1 (LLM 직접 생성)",
        "n_groups": len(persona_counts),
        "sample_size": sample_size,
        "total_population": total_users,
        "coverage_of_sample": round(len(ok_labels) / sample_size, 4),
        "coverage_of_total_population": round(len(ok_labels) / total_users, 4),
        "entropy": round(shannon_entropy(ratios), 4),
        "normalized_entropy": round(normalized_entropy(ratios), 4),
        "top1_concentration": round(top_k_concentration(ratios, 1), 4),
        "top2_concentration": round(top_k_concentration(ratios, 2), 4),
        "min_group_ratio": round(float(ratios.min()), 4),
        "max_group_ratio": round(float(ratios.max()), 4),
        "avg_labels_per_user": round(avg_personas_per_labeled_user, 4),
        "note": "multi-label(유저당 최대 5개 persona) — ratio는 전체 배정 수 기준 정규화",
    }


def compute_v2_metrics() -> dict:
    """v2 KMeans segment 배정 결과(단일 라벨)의 분포 품질 지표를 계산한다."""
    df = pd.read_csv(V2_CUSTOMER_SEGMENTS_PATH)
    total_customers = len(df)

    counts = df["segment_id"].value_counts()
    ratios = counts / total_customers

    return {
        "method": "v2 (KMeans + LLM naming)",
        "n_groups": len(counts),
        "sample_size": total_customers,
        "total_population": total_customers,
        "coverage_of_sample": 1.0,
        "coverage_of_total_population": 1.0,
        "entropy": round(shannon_entropy(ratios), 4),
        "normalized_entropy": round(normalized_entropy(ratios), 4),
        "top1_concentration": round(top_k_concentration(ratios, 1), 4),
        "top2_concentration": round(top_k_concentration(ratios, 2), 4),
        "min_group_ratio": round(float(ratios.min()), 4),
        "max_group_ratio": round(float(ratios.max()), 4),
        "avg_labels_per_user": 1.0,
        "note": "single-label(고객당 segment_id 1개) — KMeans predict()로 전원 배정",
    }


def main() -> None:
    v1_metrics = compute_v1_metrics()
    v2_metrics = compute_v2_metrics()

    result = pd.DataFrame([v1_metrics, v2_metrics])
    result.to_csv(OUTPUT_PATH, index=False)

    logger.info("[v1 vs v2 비교]\n%s", result.drop(columns=["note"]).to_string(index=False))
    logger.info("저장: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
