import json
import math
from pathlib import Path
from typing import Any

MODEL_PARAMS_PATH = Path(__file__).resolve().parent / "model_params.json"


def _load_model_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _derive_features(payload: dict, use_log_transform: bool) -> dict:
    duration_base = float(payload["foreground_app_duration_sum"])
    switch_base = float(payload["foreground_app_switch_per_hour"])
    concentration_ratio = float(payload["concentration_ratio"])

    if use_log_transform:
        # Core math: log1p transform to stabilize skewed input values.
        duration_base = math.log1p(duration_base)
        switch_base = math.log1p(switch_base)

    # Core math: derived features from duration/switch weighted by concentration ratio.
    focus_intensity = duration_base * concentration_ratio
    switch_frequency = switch_base * (1.0 - concentration_ratio)

    return {
        "focus_intensity": float(focus_intensity),
        "switch_frequency": float(switch_frequency),
    }


def _minmax_scale(value: float, data_min: float, data_max: float) -> float:
    # Core math: manual MinMax scaling without sklearn.
    denom = data_max - data_min
    if denom == 0:
        return 0.0
    return (value - data_min) / denom


def _assign_cluster(scaled_vector: list[float], centroids: list[list[float]]) -> tuple[int, float]:
    best_idx = -1
    best_distance = float("inf")

    for idx, centroid in enumerate(centroids):
        dx = scaled_vector[0] - float(centroid[0])
        dy = scaled_vector[1] - float(centroid[1])
        # Core math: Euclidean distance to each centroid.
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < best_distance:
            best_distance = distance
            best_idx = idx

    return best_idx, best_distance


def infer_cluster(payload: dict, params_path: Path | None = None) -> dict:
    params_path = params_path or MODEL_PARAMS_PATH
    params = _load_model_params(params_path)

    features = _derive_features(payload, params["log_transform_applied"])
    data_min = params["scaler"]["data_min"]
    data_max = params["scaler"]["data_max"]

    scaled_focus = _minmax_scale(features["focus_intensity"], data_min[0], data_max[0])
    scaled_switch = _minmax_scale(
        features["switch_frequency"], data_min[1], data_max[1]
    )

    assigned_cluster_id, distance_to_centroid = _assign_cluster(
        [scaled_focus, scaled_switch], params["centroids"]
    )

    return {
        "week_start": payload["week_start"],
        "week_end": payload["week_end"],
        "is_valid_data": payload["is_valid_data"],
        "features": features,
        "clustering_result": {
            "assigned_cluster_id": int(assigned_cluster_id),
            "distance_to_centroid": float(distance_to_centroid),
        },
    }
