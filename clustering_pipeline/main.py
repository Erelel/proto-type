"""
c:/Users/username/Desktop/folder/qwer/.venv/Scripts/python.exe clustering_pipeline/main.py --csv mendeley_v5.csv --k 4 --out-dir result/<MMDD>_image
"""

import argparse
import pandas as pd
from pathlib import Path

from constants import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_MAIN_CSV,
    DEFAULT_N_CLUSTERS,
    MINMAX_FEATURE_RANGE,
    resolve_project_path,
)
from preprocessing import (
    build_minmax_pipeline,
    load_and_preprocess,
)
from modeling import fit_cluster_model
from visualization import (
    plot_derived_feature_histograms,
    plot_kmeans_clusters_on_derived_features,
    plot_minmax_scaled_derived_histograms,
)


def main(csv_path: Path, n_clusters: int, output_dir: Path) -> None:
    csv_path = resolve_project_path(csv_path)
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, _, metadata = load_and_preprocess(csv_path)

    X_minmax_scaled, minmax_scaler = build_minmax_pipeline(X)
    minmax_kmeans = fit_cluster_model(X_minmax_scaled, "kmeans", n_clusters)
    minmax_centroids_original = minmax_scaler.inverse_transform(
        minmax_kmeans["model"].cluster_centers_
    )

    derived_hist_output = output_dir / "derived_feature_histograms.png"
    minmax_hist_output = output_dir / "derived_feature_histograms_minmax.png"
    cluster_output = output_dir / "kmeans_force_switch_clusters.png"
    metrics_output = output_dir / "kmeans_metrics.csv"

    plot_derived_feature_histograms(X_derived=X, output_path=derived_hist_output)
    plot_minmax_scaled_derived_histograms(
        X_derived=X,
        X_minmax_scaled=X_minmax_scaled,
        output_path=minmax_hist_output,
    )
    plot_kmeans_clusters_on_derived_features(
        X_derived=X,
        labels=minmax_kmeans["labels"],
        output_path=cluster_output,
        centroids=minmax_centroids_original,
    )

    results = pd.DataFrame(
        [
            {
                "scaler": "minmax",
                "feature_range": MINMAX_FEATURE_RANGE,
                "model": minmax_kmeans["model_name"],
                "n_clusters": n_clusters,
                "silhouette": minmax_kmeans["silhouette"],
                "davies_bouldin": minmax_kmeans["davies_bouldin"],
            },
        ]
    )
    results.to_csv(metrics_output, index=False)

    print("=== Preprocessing Summary ===")
    print(f"Initial rows: {metadata['initial_rows']}")
    print(f"Rows after dropna: {metadata['final_rows']}")
    print(f"Dropped rows (NaN): {metadata['dropped_nan_rows']}")
    print(f"Used derived features: {metadata['used_columns']}")
    print(f"Source skewness: {metadata['source_skewness']}")
    print(f"Skew direction: {metadata['skew_direction']}")
    print(f"Applied log transform in derived features: {metadata['log_transform_applied']}")
    if not metadata["log_transform_applied"]:
        print(f"Reason log transform was skipped: {metadata['log_transform_block_reason']}")

    print("\n=== KMeans Metrics (MinMax only) ===")
    print(results.to_string(index=False))

    print("\n=== Saved Outputs ===")
    print(f"Metrics CSV: {metrics_output}")
    print(f"Derived histogram image: {derived_hist_output}")
    print(f"MinMax histogram image: {minmax_hist_output}")
    print(f"KMeans scatter image: {cluster_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMeans pipeline with MinMax scaling")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_MAIN_CSV,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Number of clusters",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory where plots and metrics will be saved",
    )
    args = parser.parse_args()

    main(args.csv, args.k, args.out_dir)
