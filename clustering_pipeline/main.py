"""
c:/Users/username/Desktop/folder/qwer/.venv/Scripts/python.exe clustering_pipeline/main.py --csv mendeley_v4.csv --k 3 --out-dir result/0402_image
"""

import argparse
import pandas as pd
from pathlib import Path

from constants import DEFAULT_N_CLUSTERS
from preprocessing import (
    build_raw_pipeline,
    build_standard_pipeline,
    load_and_preprocess,
)
from modeling import fit_cluster_model
from visualization import (
    plot_minmax_vs_standard_kmeans_compare,
    plot_kmeans_clusters_on_derived_features,
    plot_source_histograms,
)


def main(csv_path: Path, n_clusters: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    X, source_df, metadata = load_and_preprocess(csv_path)

    X_minmax_scaled, minmax_scaler = build_raw_pipeline(X)
    minmax_kmeans = fit_cluster_model(X_minmax_scaled, "kmeans", n_clusters)
    minmax_centroids_original = minmax_scaler.inverse_transform(
        minmax_kmeans["model"].cluster_centers_
    )

    X_standard_scaled, standard_scaler = build_standard_pipeline(X)
    standard_kmeans = fit_cluster_model(X_standard_scaled, "kmeans", n_clusters)
    standard_centroids_original = standard_scaler.inverse_transform(
        standard_kmeans["model"].cluster_centers_
    )

    histogram_output = output_dir / "source_histogram_skewness.png"
    cluster_output = output_dir / "kmeans_force_switch_clusters.png"
    compare_output = output_dir / "minmax_vs_standard_kmeans_compare.png"
    metrics_output = output_dir / "kmeans_metrics.csv"

    plot_source_histograms(source_df, histogram_output, metadata["source_skewness"])
    plot_kmeans_clusters_on_derived_features(
        X_derived=X,
        labels=minmax_kmeans["labels"],
        output_path=cluster_output,
        centroids=minmax_centroids_original,
    )
    plot_minmax_vs_standard_kmeans_compare(
        X_derived=X,
        labels_minmax=minmax_kmeans["labels"],
        labels_standard=standard_kmeans["labels"],
        output_path=compare_output,
        centroids_minmax=minmax_centroids_original,
        centroids_standard=standard_centroids_original,
    )

    results = pd.DataFrame(
        [
            {
                "scaler": "minmax",
                "model": minmax_kmeans["model_name"],
                "n_clusters": n_clusters,
                "silhouette": minmax_kmeans["silhouette"],
                "davies_bouldin": minmax_kmeans["davies_bouldin"],
            },
            {
                "scaler": "standard",
                "model": standard_kmeans["model_name"],
                "n_clusters": n_clusters,
                "silhouette": standard_kmeans["silhouette"],
                "davies_bouldin": standard_kmeans["davies_bouldin"],
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

    print("\n=== KMeans Metrics (MinMax vs Standard) ===")
    print(results.sort_values(by=["silhouette", "davies_bouldin"], ascending=[False, True]).to_string(index=False))

    print("\n=== Saved Outputs ===")
    print(f"Metrics CSV: {metrics_output}")
    print(f"Histogram image: {histogram_output}")
    print(f"KMeans scatter image: {cluster_output}")
    print(f"Scaler comparison image: {compare_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KMeans/MiniBatchKMeans comparison pipeline")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mendeley_v4.csv"),
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
        default=Path("result") / "0402_image",
        help="Directory where plots and metrics will be saved",
    )
    args = parser.parse_args()

    main(args.csv, args.k, args.out_dir)
