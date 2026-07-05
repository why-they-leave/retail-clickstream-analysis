"""
segment feature table을 기반으로 clustering segment_id와 segment summary를 생성한다.

Full 데이터 단일 트랙 (Issue #23 — US-only 트랙 제거).
#4에서 Full과 US-only 간 유의미한 차이가 확인되지 않아, 전체 고객(Full) 기준으로만
scaler/KMeans를 fit하고 segment_id를 부여한다.

입력:
    data/processed/segment_features_all_customers.csv

출력:
    data/processed/customer_segments_all_customers.csv
    data/processed/segment_summary_all_customers.csv
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.segment_common import fill_clustering_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CONFIG_PATH = ROOT_DIR / "configs" / "segment" / "params.yaml"

SEGMENT_FEATURE_PATH = PROCESSED_DIR / "segment_features_all_customers.csv"
CUSTOMER_SEGMENT_PATH = PROCESSED_DIR / "customer_segments_all_customers.csv"
SEGMENT_SUMMARY_PATH = PROCESSED_DIR / "segment_summary_all_customers.csv"

DEFAULT_SEGMENT_CONFIG = {
    "method": "kmeans",
    "n_clusters": 6,
    "random_state": 42,
    "n_init": 20,
    "input_features": [
        "page_view_count",
        "atc_rate",
        "order_count",
        "purchase_per_session",
        "total_spend_log",
        "recency_session_days",
        "recency_order_days",
        "view_purchase_category_match",
        "dominant_view_category_ratio",
        "dominant_purchase_category_ratio",
    ],
}


def load_segment_config(config_path: Path = CONFIG_PATH) -> dict:
    """segment assignment 설정을 YAML에서 로드한다."""
    if not config_path.exists():
        logger.warning("[config] 설정 파일이 없어 기본값을 사용합니다: %s", config_path)
        return DEFAULT_SEGMENT_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    config = DEFAULT_SEGMENT_CONFIG.copy()
    config.update(raw_config.get("segment_assignment", {}))

    if config["method"] != "kmeans":
        raise ValueError(f"지원하지 않는 segment assignment method: {config['method']}")
    if not config["input_features"]:
        raise ValueError("segment_assignment.input_features가 비어 있습니다.")

    return config


def fit_segment_model(df: pd.DataFrame, config: dict) -> Pipeline:
    """전체 고객 기준으로 scaler와 KMeans pipeline을 학습한다."""
    x = fill_clustering_features(df, df, config["input_features"])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=config["n_clusters"],
                    n_init=config["n_init"],
                    random_state=config["random_state"],
                ),
            ),
        ]
    )
    pipeline.fit(x)
    return pipeline


def assign_segment_ids(
    df: pd.DataFrame,
    pipeline: Pipeline,
    input_features: list[str],
) -> pd.DataFrame:
    """학습된 pipeline으로 customer_id별 segment_id를 부여한다."""
    x = fill_clustering_features(df, df, input_features)
    segment_ids = pipeline.predict(x)

    result = df.copy()
    result["segment_id"] = segment_ids
    return result


def _top_category_distribution(
    df: pd.DataFrame,
    segment_id: int,
    column: str,
    top_n: int = 3,
) -> str:
    """segment 내 상위 카테고리 비율을 사람이 읽기 쉬운 문자열로 반환한다."""
    segment = df[df["segment_id"] == segment_id]
    values = segment[column].dropna()
    if values.empty:
        return ""

    ratios = values.value_counts(normalize=True).head(top_n)
    return "; ".join(f"{category}: {ratio:.1%}" for category, ratio in ratios.items())


def build_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """LLM naming/summarization 입력으로 사용할 segment summary를 생성한다."""
    total_customers = len(df)

    summary = (
        df.groupby("segment_id")
        .agg(
            customer_count=("customer_id", "nunique"),
            avg_session_count=("session_count", "mean"),
            avg_page_view_count=("page_view_count", "mean"),
            avg_add_to_cart_count=("add_to_cart_count", "mean"),
            avg_atc_rate=("atc_rate", "mean"),
            purchaser_ratio=("is_purchaser", "mean"),
            avg_order_count=("order_count", "mean"),
            avg_purchase_per_session=("purchase_per_session", "mean"),
            avg_total_spend_log=("total_spend_log", "mean"),
            median_total_spend_log=("total_spend_log", "median"),
            avg_avg_order_value_log=("avg_order_value_log", "mean"),
            avg_recency_session_days=("recency_session_days", "mean"),
            avg_recency_order_days=("recency_order_days", "mean"),
            avg_category_diversity_purchase=("category_diversity_purchase", "mean"),
            avg_dominant_view_category_ratio=("dominant_view_category_ratio", "mean"),
            avg_dominant_purchase_category_ratio=("dominant_purchase_category_ratio", "mean"),
            view_purchase_category_match_rate=("view_purchase_category_match", "mean"),
        )
        .reset_index()
        .sort_values("segment_id")
    )
    summary["customer_ratio"] = summary["customer_count"] / total_customers
    summary["top_view_categories"] = summary["segment_id"].apply(
        lambda segment_id: _top_category_distribution(df, segment_id, "top_view_category")
    )
    summary["top_purchase_categories"] = summary["segment_id"].apply(
        lambda segment_id: _top_category_distribution(df, segment_id, "top_purchase_category")
    )

    ordered_cols = [
        "segment_id",
        "customer_count",
        "customer_ratio",
        "avg_session_count",
        "avg_page_view_count",
        "avg_add_to_cart_count",
        "avg_atc_rate",
        "purchaser_ratio",
        "avg_order_count",
        "avg_purchase_per_session",
        "avg_total_spend_log",
        "median_total_spend_log",
        "avg_avg_order_value_log",
        "avg_recency_session_days",
        "avg_recency_order_days",
        "avg_category_diversity_purchase",
        "avg_dominant_view_category_ratio",
        "avg_dominant_purchase_category_ratio",
        "view_purchase_category_match_rate",
        "top_view_categories",
        "top_purchase_categories",
    ]
    return summary[ordered_cols]


def validate_assignments(df: pd.DataFrame, label: str, n_clusters: int) -> None:
    """segment assignment 결과의 기본 무결성을 검증한다."""
    errors = []
    if df["customer_id"].duplicated().any():
        errors.append("customer_id 중복 존재")
    if df["segment_id"].isna().any():
        errors.append("segment_id 결측 존재")
    if df["segment_id"].nunique() > n_clusters:
        errors.append("segment_id 개수가 설정한 cluster 수보다 큼")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"[{label}] segment assignment 검증 실패:\n{details}")
    logger.info(
        "[%s] segment assignment 검증 통과 (rows=%s, segments=%s)",
        label,
        f"{len(df):,}",
        df["segment_id"].nunique(),
    )


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config = load_segment_config()
    logger.info(
        "[config] method=%s, n_clusters=%s, random_state=%s, n_init=%s",
        config["method"],
        config["n_clusters"],
        config["random_state"],
        config["n_init"],
    )

    df = pd.read_csv(SEGMENT_FEATURE_PATH)
    pipeline = fit_segment_model(df, config)
    result = assign_segment_ids(df, pipeline, config["input_features"])
    validate_assignments(result, "all_customers", config["n_clusters"])

    result.to_csv(CUSTOMER_SEGMENT_PATH, index=False)
    logger.info("[all_customers] 저장: %s", CUSTOMER_SEGMENT_PATH)

    summary = build_segment_summary(result)
    summary.to_csv(SEGMENT_SUMMARY_PATH, index=False)
    logger.info("[all_customers] 저장: %s", SEGMENT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
