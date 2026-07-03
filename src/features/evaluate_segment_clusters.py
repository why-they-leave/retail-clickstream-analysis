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
from segment_common import fill_clustering_features
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SEGMENT_FEATURE_PATH = PROCESSED_DIR / "segment_features_all_customers.csv"
EVALUATION_PATH = PROCESSED_DIR / "segment_cluster_evaluation.csv"
SIZE_PATH = PROCESSED_DIR / "segment_cluster_sizes.csv"

K_RANGE = range(3, 10)


def evaluate_k_candidates(
    df: pd.DataFrame,
    input_features: list[str],
    k_range=K_RANGE,
    n_init: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """후보 k별 inertia, silhouette, segment size balance를 계산한다."""
    x = fill_clustering_features(df, df, input_features)

    evaluation_records = []
    size_records = []

    for k in k_range:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=random_state)),
            ]
        )
        labels = pipeline.fit_predict(x)
        x_scaled = pipeline.named_steps["scaler"].transform(x)
        kmeans = pipeline.named_steps["kmeans"]

        label_counts = pd.Series(labels, name="segment_id").value_counts().sort_index()
        label_ratios = label_counts / len(labels)
        silhouette = silhouette_score(x_scaled, labels)

        evaluation_records.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
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
