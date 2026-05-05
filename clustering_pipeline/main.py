"""
c:/Users/username/Desktop/folder/qwer/.venv/Scripts/python.exe clustering_pipeline/main.py --csv mendeley_v5.csv --k 4 --out-dir result/<MMDD>_image
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import davies_bouldin_score, silhouette_score

from constants import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_MAIN_CSV,
    DEFAULT_N_CLUSTERS,
    MINMAX_FEATURE_RANGE,
    PIPELINE_DIR,
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


def _build_cluster_distribution(labels: np.ndarray, dataset: str) -> pd.DataFrame:
    labels_array = np.asarray(labels, dtype=int)
    if labels_array.size == 0:
        return pd.DataFrame(columns=["dataset", "cluster", "count", "ratio"])

    cluster_distribution = (
        pd.Series(labels_array, name="cluster")
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    cluster_distribution["ratio"] = (
        cluster_distribution["count"] / cluster_distribution["count"].sum()
    )
    cluster_distribution.insert(0, "dataset", dataset)
    return cluster_distribution


def _compute_quality_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, str]:
    labels_array = np.asarray(labels, dtype=int)
    unique_labels = np.unique(labels_array)

    if len(unique_labels) < 2:
        return (
            float("nan"),
            float("nan"),
            "Silhouette/DBI skipped because fewer than 2 clusters were predicted.",
        )

    silhouette = float(silhouette_score(X_scaled, labels_array))
    davies_bouldin = float(davies_bouldin_score(X_scaled, labels_array))
    return silhouette, davies_bouldin, ""


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.6f}"


def main(
    csv_path: Path,
    n_clusters: int,
    output_dir: Path,
    collected_csv_path: Path | None = None,
) -> None:
    csv_path = resolve_project_path(csv_path)
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if collected_csv_path is not None:
        collected_csv_path = resolve_project_path(collected_csv_path)

    X_train, _, train_metadata = load_and_preprocess(csv_path)

    X_minmax_scaled, minmax_scaler = build_minmax_pipeline(X_train)
    minmax_kmeans = fit_cluster_model(X_minmax_scaled, "kmeans", n_clusters)
    minmax_centroids_original = minmax_scaler.inverse_transform(
        minmax_kmeans["model"].cluster_centers_
    )

    model_params = {
        "scaler": {
            "data_min": minmax_scaler.data_min_.tolist(),
            "data_max": minmax_scaler.data_max_.tolist(),
        },
        "centroids": minmax_kmeans["model"].cluster_centers_.tolist(),
        "log_transform_applied": train_metadata["log_transform_applied"],
    }
    params_output = PIPELINE_DIR / "model_params.json"
    with params_output.open("w", encoding="utf-8") as file:
        json.dump(model_params, file, ensure_ascii=True, indent=2)

    derived_hist_output = output_dir / "derived_feature_histograms.png"
    minmax_hist_output = output_dir / "derived_feature_histograms_minmax.png"
    cluster_output = output_dir / "kmeans_focus_switch_clusters.png"
    collected_cluster_output = output_dir / "kmeans_focus_switch_clusters_collected.png"
    metrics_output = output_dir / "kmeans_metrics.csv"
    cluster_distribution_output = output_dir / "kmeans_cluster_distribution.csv"

    plot_derived_feature_histograms(X_derived=X_train, output_path=derived_hist_output)
    plot_minmax_scaled_derived_histograms(
        X_derived=X_train,
        X_minmax_scaled=X_minmax_scaled,
        output_path=minmax_hist_output,
    )
    plot_kmeans_clusters_on_derived_features(
        X_derived=X_train,
        labels=minmax_kmeans["labels"],
        output_path=cluster_output,
        centroids=minmax_centroids_original,
    )

    metrics_rows: list[dict] = [
        {
            "dataset": "baseline_train",
            "source_csv": csv_path.name,
            "n_rows": train_metadata["final_rows"],
            "scaler": "minmax",
            "feature_range": MINMAX_FEATURE_RANGE,
            "model": minmax_kmeans["model_name"],
            "n_clusters": n_clusters,
            "silhouette": minmax_kmeans["silhouette"],
            "davies_bouldin": minmax_kmeans["davies_bouldin"],
            "note": "",
        }
    ]
    cluster_distribution_frames = [
        _build_cluster_distribution(minmax_kmeans["labels"], dataset="baseline_train")
    ]

    collected_metadata = None
    if collected_csv_path is not None:
        X_collected, _, collected_metadata = load_and_preprocess(
            collected_csv_path,
            force_log_transform=train_metadata["log_transform_applied"],
        )
        if X_collected.empty:
            raise ValueError("Collected CSV has no valid rows after preprocessing.")

        X_collected_scaled = minmax_scaler.transform(X_collected)
        collected_labels = minmax_kmeans["model"].predict(X_collected_scaled)
        collected_silhouette, collected_davies_bouldin, collected_note = (
            _compute_quality_metrics(X_collected_scaled, collected_labels)
        )

        metrics_rows.append(
            {
                "dataset": "collected_eval",
                "source_csv": collected_csv_path.name,
                "n_rows": collected_metadata["final_rows"],
                "scaler": "minmax(frozen_from_baseline)",
                "feature_range": MINMAX_FEATURE_RANGE,
                "model": minmax_kmeans["model_name"],
                "n_clusters": n_clusters,
                "silhouette": collected_silhouette,
                "davies_bouldin": collected_davies_bouldin,
                "note": collected_note,
            }
        )
        cluster_distribution_frames.append(
            _build_cluster_distribution(collected_labels, dataset="collected_eval")
        )
        plot_kmeans_clusters_on_derived_features(
            X_derived=X_collected,
            labels=collected_labels,
            output_path=collected_cluster_output,
            centroids=minmax_centroids_original,
        )

    results = pd.DataFrame(metrics_rows)
    cluster_distribution = pd.concat(
        cluster_distribution_frames,
        ignore_index=True,
    )
    results.to_csv(metrics_output, index=False)
    cluster_distribution.to_csv(cluster_distribution_output, index=False)

    print("=== Preprocessing Summary (Baseline train) ===")
    print(f"Source CSV: {csv_path}")
    print(f"Initial rows: {train_metadata['initial_rows']}")
    print(f"Rows after dropna: {train_metadata['final_rows']}")
    print(f"Dropped rows (NaN): {train_metadata['dropped_nan_rows']}")
    print(f"Used derived features: {train_metadata['used_columns']}")
    print(f"Source skewness: {train_metadata['source_skewness']}")
    print(f"Skew direction: {train_metadata['skew_direction']}")
    print(
        f"Applied log transform in derived features: "
        f"{train_metadata['log_transform_applied']}"
    )
    if not train_metadata["log_transform_applied"]:
        print(
            "Reason log transform was skipped: "
            f"{train_metadata['log_transform_block_reason']}"
        )

    if collected_metadata is not None and collected_csv_path is not None:
        print("\n=== Preprocessing Summary (Collected eval) ===")
        print(f"Source CSV: {collected_csv_path}")
        print(f"Initial rows: {collected_metadata['initial_rows']}")
        print(f"Rows after dropna: {collected_metadata['final_rows']}")
        print(f"Dropped rows (NaN): {collected_metadata['dropped_nan_rows']}")
        print(f"Used derived features: {collected_metadata['used_columns']}")
        print(f"Source skewness: {collected_metadata['source_skewness']}")
        print(f"Skew direction: {collected_metadata['skew_direction']}")
        print(
            "Applied log transform in derived features: "
            f"{collected_metadata['log_transform_applied']} "
            "(forced to baseline rule)"
        )

    print("\n=== KMeans Metrics ===")
    print(
        results.to_string(
            index=False,
            formatters={
                "silhouette": _format_metric,
                "davies_bouldin": _format_metric,
            },
        )
    )

    print("\n=== Cluster Distribution ===")
    print(
        cluster_distribution.to_string(
            index=False,
            formatters={"ratio": lambda value: f"{value:.4f}"},
        )
    )

    print("\n=== Saved Outputs ===")
    print(f"Metrics CSV: {metrics_output}")
    print(f"Cluster distribution CSV: {cluster_distribution_output}")
    print(f"Derived histogram image: {derived_hist_output}")
    print(f"MinMax histogram image: {minmax_hist_output}")
    print(f"Model params JSON: {params_output}")
    print(f"KMeans scatter image (baseline): {cluster_output}")
    if collected_csv_path is not None:
        print(f"KMeans scatter image (collected): {collected_cluster_output}")


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
    parser.add_argument(
        "--collected-csv",
        type=Path,
        default=None,
        help=(
            "Optional collected CSV path. "
            "The model trained on --csv is frozen and evaluated on this dataset."
        ),
    )
    args = parser.parse_args()

    main(args.csv, args.k, args.out_dir, args.collected_csv)
