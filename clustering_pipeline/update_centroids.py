import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from constants import DEFAULT_MAIN_CSV, DEFAULT_N_CLUSTERS, PIPELINE_DIR, resolve_project_path
from modeling import fit_cluster_model
from preprocessing import build_minmax_pipeline, load_and_preprocess


MODEL_PARAMS_PATH = PIPELINE_DIR / "model_params.json"


def _load_model_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_model_params(path: Path, params: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=True, indent=2)


def _reorder_centroids(
    old_centroids: np.ndarray,
    new_centroids: np.ndarray,
) -> np.ndarray:
    # Core math: cost matrix of Euclidean distances between old and new centroids.
    cost_matrix = cdist(old_centroids, new_centroids, metric="euclidean")
    # Core math: Hungarian algorithm for minimum-cost 1:1 assignment.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    reordered = np.zeros_like(new_centroids)
    for old_idx, new_idx in zip(row_ind, col_ind):
        reordered[old_idx] = new_centroids[new_idx]

    return reordered


def update_centroids(
    csv_path: Path,
    n_clusters: int,
    params_path: Path = MODEL_PARAMS_PATH,
) -> None:
    csv_path = resolve_project_path(csv_path)
    params_path = resolve_project_path(params_path)

    X_train, _, train_metadata = load_and_preprocess(csv_path)
    X_minmax_scaled, minmax_scaler = build_minmax_pipeline(X_train)
    kmeans_result = fit_cluster_model(X_minmax_scaled, "kmeans", n_clusters)

    new_centroids = np.asarray(kmeans_result["model"].cluster_centers_, dtype=float)
    params = _load_model_params(params_path)
    old_centroids = np.asarray(params["centroids"], dtype=float)

    if old_centroids.shape != new_centroids.shape:
        raise ValueError(
            "Centroid shape mismatch between old and new models. "
            f"old={old_centroids.shape}, new={new_centroids.shape}"
        )

    reordered_centroids = _reorder_centroids(old_centroids, new_centroids)

    params["centroids"] = reordered_centroids.tolist()
    params["scaler"] = {
        "data_min": minmax_scaler.data_min_.tolist(),
        "data_max": minmax_scaler.data_max_.tolist(),
    }
    params["log_transform_applied"] = train_metadata["log_transform_applied"]

    _save_model_params(params_path, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update KMeans centroids with Hungarian matching"
    )
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
        "--params-path",
        type=Path,
        default=MODEL_PARAMS_PATH,
        help="Path to model_params.json",
    )

    args = parser.parse_args()
    update_centroids(args.csv, args.k, args.params_path)
