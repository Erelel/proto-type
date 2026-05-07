from __future__ import annotations

from copy import deepcopy
from typing import Any


def _coerce_centroid_row(row: Any) -> list[float]:
    try:
        return [float(value) for value in row]
    except TypeError as exc:
        raise ValueError("Each centroid row must be an iterable of numeric values.") from exc


def restore_centroids_from_storage(centroids: Any) -> list[list[float]]:
    if isinstance(centroids, dict):
        try:
            ordered_items = sorted(centroids.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Firestore centroids must use integer-like string keys such as '0', '1', '2'."
            ) from exc
        return [_coerce_centroid_row(row) for _, row in ordered_items]

    if isinstance(centroids, list):
        return [_coerce_centroid_row(row) for row in centroids]

    raise ValueError("Centroids must be stored as either a list or a Firestore-safe dict.")


def normalize_model_params(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("Model params must be a dictionary.")

    normalized = deepcopy(params)
    normalized["centroids"] = restore_centroids_from_storage(
        normalized.get("centroids")
    )
    return normalized


def serialize_centroids_for_firestore(centroids: Any) -> dict[str, list[float]]:
    restored = restore_centroids_from_storage(centroids)
    return {str(idx): row for idx, row in enumerate(restored)}


def serialize_model_params_for_firestore(params: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_model_params(params)
    normalized["centroids"] = serialize_centroids_for_firestore(
        normalized["centroids"]
    )
    return normalized
