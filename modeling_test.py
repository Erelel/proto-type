import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from constants import RANDOM_STATE
"""

1. Phase 1: Day 0 (서비스 런칭 및 실시간 추론 단계)
이 단계(모바일/서버의 실시간 예측)에서는 절대 fit을 사용하지 않습니다.

동작 원리: 사전 학습되어 동결된(Frozen) 베이스라인 모델의 파라미터만 사용합니다.
linear_sum_assignment 위치: 사용하지 않습니다. 
과거 모델과 비교할 '새로운 중심점'이 존재하지 않으므로 매칭 알고리즘이 개입할 이유가 없습니다.
inverse_transform 위치: 모델링 직후가 아니라, LLM 진단 멘트 호출 직전에 사용합니다. 
스케일링된 '유저 개인의 데이터'를 원래 수치(분, 횟수)로 복원하여 프롬프트에 절대값으로 쏴주기 위한 용도로만 쓰입니다.

2. Phase 2: Day 7 이후 (데이터 누적 후 오프라인 재학습 단계)
이 단계(백엔드 스케줄러에 의한 모델 갱신)에서는 새로운 중심점을 찾아야 하므로 반드시 fit을 사용해야 합니다.
앞서 언급된 두 함수의 논리적 위치는 이 Phase 2에 정확히 해당합니다.

동작 원리: 유저 실데이터가 충분히 누적되면 백그라운드에서 새로운 K-Means 모델을 학습(fit)시킵니다.
inverse_transform 위치 (중심점 해석용): fit이 끝나 새로운 중심점이 도출된 직후에 위치합니다. 이 중심점들을 원래 단위로 복원하여, 전체 생태계의 사용량이 상향 평준화되었는지 검증합니다.
linear_sum_assignment 위치 (동적 계승용): 재학습 파이프라인의 가장 마지막에 위치합니다. Day 0의 '과거 중심점'과 Day 7의 '새로운 중심점' 간의 거리 행렬(Cost Matrix)을 계산하여 라벨(Label ID)을 덮어씌웁니다.


"""

def predict_with_frozen_model(X_scaled: np.ndarray, frozen_model) -> np.ndarray:
    """Phase 1: Day 0 real-time inference (no fit)."""
    return frozen_model.predict(X_scaled)


def inverse_transform_for_prompt(user_scaled: np.ndarray, scaler) -> np.ndarray:
    """Phase 1: restore user-level scaled values right before LLM prompt generation."""
    user_scaled_2d = np.atleast_2d(user_scaled)
    restored = scaler.inverse_transform(user_scaled_2d)
    if np.ndim(user_scaled) == 1:
        return restored[0]
    return restored


def _align_new_centroids_to_previous(
    previous_centroids_scaled: np.ndarray,
    new_centroids_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase 2 final step: align new cluster IDs to previous IDs via Hungarian matching.

    Returns:
        reordered_new_centroids: new centroids reordered to previous ID order
        old_idx: indices of previous centroids selected by linear_sum_assignment
        new_idx: indices of new centroids matched to old_idx
    """
    if previous_centroids_scaled.shape != new_centroids_scaled.shape:
        raise ValueError(
            "previous_centroids_scaled and new_centroids_scaled must have the same shape."
        )

    cost_matrix = pairwise_distances(previous_centroids_scaled, new_centroids_scaled)
    old_idx, new_idx = linear_sum_assignment(cost_matrix)
    reordered_new_centroids = np.empty_like(new_centroids_scaled)
    reordered_new_centroids[old_idx] = new_centroids_scaled[new_idx]
    return reordered_new_centroids, old_idx, new_idx


def fit_cluster_model(
    X_scaled: np.ndarray,
    model_name: str,
    n_clusters: int,
    scaler=None,
    previous_centroids_scaled: np.ndarray | None = None,
) -> dict:
    """Phase 2: offline retraining (fit required)."""
    if model_name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
        display_name = "KMeans"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    labels = model.fit_predict(X_scaled)
    new_centroids_scaled = model.cluster_centers_

    if len(np.unique(labels)) < 2:
        raise ValueError(f"{display_name} produced fewer than 2 clusters.")

    # Phase 2: interpret freshly trained centroids in original units right after fit.
    centroids_original = None
    if scaler is not None:
        centroids_original = scaler.inverse_transform(new_centroids_scaled)

    label_mapping = None
    if previous_centroids_scaled is not None:
        # Phase 2 final step: overwrite label IDs to inherit previous cluster semantics.
        aligned_centroids, old_idx, new_idx = _align_new_centroids_to_previous(
            previous_centroids_scaled, new_centroids_scaled
        )
        id_map = np.full(n_clusters, -1, dtype=int)
        id_map[new_idx] = old_idx
        labels = id_map[labels]
        model.cluster_centers_ = aligned_centroids
        label_mapping = {
            int(new_id): int(old_id) for old_id, new_id in zip(old_idx, new_idx)
        }

    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    return {
        "model": model,
        "labels": labels,
        "model_name": display_name,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "centroids_scaled": model.cluster_centers_,
        "centroids_original": centroids_original,
        "label_mapping": label_mapping,
    }


def _minmax_scale(value: float, min_val: float, max_val: float) -> float:
    """Manual MinMax scaling with safe zero-division handling."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def predict_cluster(raw_data: dict, saved_params: dict) -> int:
    """
    Stateless inference using only saved parameters.

    Steps:
        1) Derived feature calculation (+ optional log1p)
        2) Manual MinMax scaling via saved scaler params
        3) Nearest centroid mapping with Euclidean distance
    """
    duration_base = raw_data["foreground_app_duration_sum"]
    switch_base = raw_data["foreground_app_switch_per_hour"]

    if saved_params.get("log_transform_applied", False):
        duration_base = np.log1p(duration_base)
        switch_base = np.log1p(switch_base)

    concentration_ratio = raw_data["concentration_ratio"]
    focus_intensity = duration_base * concentration_ratio
    switch_frequency = switch_base * (1.0 - concentration_ratio)

    scaler_params = saved_params["scaler_params"]
    focus_scaled = _minmax_scale(
        focus_intensity,
        scaler_params["focus_intensity"]["min"],
        scaler_params["focus_intensity"]["max"],
    )
    switch_scaled = _minmax_scale(
        switch_frequency,
        scaler_params["switch_frequency"]["min"],
        scaler_params["switch_frequency"]["max"],
    )

    vector = np.array([focus_scaled, switch_scaled], dtype=float)
    centroids = np.array(saved_params["centroids"], dtype=float)
    distances = np.linalg.norm(centroids - vector, axis=1)
    return int(np.argmin(distances))


def align_clusters(
    old_centroids: np.ndarray, new_centroids: np.ndarray
) -> tuple[dict[int, int], np.ndarray]:
    """
    Align newly trained clusters to previous cluster IDs via Hungarian algorithm.

    Returns:
        mapping: {new_label: old_label}
        reordered_new_centroids: new centroids in old-label order
    """
    if old_centroids.shape != new_centroids.shape:
        raise ValueError("old_centroids and new_centroids must have the same shape.")

    cost_matrix = cdist(old_centroids, new_centroids)
    old_idx, new_idx = linear_sum_assignment(cost_matrix)

    mapping = {int(new_id): int(old_id) for old_id, new_id in zip(old_idx, new_idx)}
    reordered_new_centroids = np.empty_like(new_centroids)
    reordered_new_centroids[old_idx] = new_centroids[new_idx]
    return mapping, reordered_new_centroids


if __name__ == "__main__":
    # Mock data for Cloud Function usage simulation.
    raw_data_example = {
        "foreground_app_duration_sum": 180.0,
        "foreground_app_switch_per_hour": 12.0,
        "concentration_ratio": 0.65,
    }
    saved_params_example = {
        "log_transform_applied": True,
        "scaler_params": {
            "focus_intensity": {"min": 0.0, "max": 4.5},
            "switch_frequency": {"min": 0.0, "max": 3.0},
        },
        "centroids": [
            [0.1, 0.2],
            [0.3, 0.7],
            [0.8, 0.4],
            [0.9, 0.9],
        ],
    }

    predicted = predict_cluster(raw_data_example, saved_params_example)
    print("Predicted cluster:", predicted)

    old_c = np.array([[0.1, 0.2], [0.4, 0.8], [0.7, 0.5], [0.9, 0.9]])
    new_c = np.array([[0.42, 0.79], [0.12, 0.22], [0.88, 0.92], [0.68, 0.52]])
    mapping, reordered = align_clusters(old_c, new_c)
    print("Mapping (new -> old):", mapping)
    print("Reordered centroids:\n", reordered)