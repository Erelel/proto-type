import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from constants import RANDOM_STATE


def fit_cluster_model(X_scaled: np.ndarray, model_name: str, n_clusters: int) -> dict:
    if model_name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
        display_name = "KMeans"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    labels = model.fit_predict(X_scaled)

    if len(np.unique(labels)) < 2:
        raise ValueError(f"{display_name} produced fewer than 2 clusters.")

    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)

    return {
        "model": model,
        "labels": labels,
        "model_name": display_name,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
    }