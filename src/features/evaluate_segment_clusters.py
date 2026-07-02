"""
KMeans segment assignment의 후보 cluster 수를 비교한다.

입력:
    data/processed/segment_features_all_customers.csv

출력:
    data/processed/segment_cluster_evaluation.csv
    data/processed/segment_cluster_sizes.csv
"""

import logging
from pathlib import Path

import pandas as pd
from assign_segments import load_segment_config
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SEGMENT_FEATURE_PATH = PROCESSED_DIR / "segment_features_all_customers.csv"
EVALUATION_PATH = PROCESSED_DIR / "segment_cluster_evaluation.csv"
SIZE_PATH = PROCESSED_DIR / "segment_cluster_sizes.csv"

K_RANGE = range(3, 10)


def _fill_clustering_features(
    df: pd.DataFrame,
    input_features: list[str],
) -> pd.DataFrame:
    """assign_segments.py와 같은 기준으로 cluster input 결측을 대체한다."""
    missing_cols = sorted(set(input_features) - set(df.columns))
    if missing_cols:
        raise ValueError(f"segment feature table에 필요한 컬럼이 없습니다: {missing_cols}")

    result = df[input_features].copy()

    zero_fill_cols = [
        "page_view_count",
        "atc_rate",
        "order_count",
        "purchase_per_session",
        "total_spend_log",
        "view_purchase_category_match",
        "dominant_view_category_ratio",
        "dominant_purchase_category_ratio",
    ]
    zero_fill_cols = [col for col in zero_fill_cols if col in result.columns]
    result[zero_fill_cols] = result[zero_fill_cols].fillna(0)

    for col in ["recency_session_days", "recency_order_days"]:
        if col not in result.columns:
            continue
        max_value = result[col].dropna().max()
        fill_value = max_value + 1 if pd.notna(max_value) else 0
        result[col] = result[col].fillna(fill_value)

    return result


def evaluate_k_candidates(
    df: pd.DataFrame,
    input_features: list[str],
    k_range=K_RANGE,
    n_init: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """후보 k별 inertia, silhouette, segment size balance를 계산한다."""
    x = _fill_clustering_features(df, input_features)
    x_scaled = StandardScaler().fit_transform(x)

    evaluation_records = []
    size_records = []

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(x_scaled)

        label_counts = pd.Series(labels, name="segment_id").value_counts().sort_index()
        label_ratios = label_counts / len(labels)
        silhouette = silhouette_score(x_scaled, labels)

        evaluation_records.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette,
                "min_segment_count": int(label_counts.min()),
                "max_segment_count": int(label_counts.max()),
                "min_segment_ratio": float(label_ratios.min()),
                "max_segment_ratio": float(label_ratios.max()),
            }
        )

        for segment_id, count in label_counts.items():
            size_records.append(
                {
                    "k": k,
                    "segment_id": int(segment_id),
                    "customer_count": int(count),
                    "customer_ratio": float(count / len(labels)),
                }
            )

    return pd.DataFrame(evaluation_records), pd.DataFrame(size_records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    config = load_segment_config()
    df = pd.read_csv(SEGMENT_FEATURE_PATH)

    evaluation, sizes = evaluate_k_candidates(
        df,
        input_features=config["input_features"],
        k_range=K_RANGE,
        n_init=config["n_init"],
        random_state=config["random_state"],
    )

    evaluation.to_csv(EVALUATION_PATH, index=False)
    sizes.to_csv(SIZE_PATH, index=False)

    logger.info("[cluster evaluation]\n%s", evaluation.round(4).to_string(index=False))
    logger.info("[저장] %s", EVALUATION_PATH)
    logger.info("[저장] %s", SIZE_PATH)


if __name__ == "__main__":
    main()
