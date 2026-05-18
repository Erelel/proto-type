from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# [FIXED] Use a scalable Hungarian assignment implementation for centroid alignment.
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    # [FIXED] DBI is logged as a reference-only metric, never used for K selection.
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / "clustering_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from preprocessing import build_minmax_pipeline, load_and_preprocess


OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "gd_05181312.csv"

RESULT_JSON_PATH = OUTPUT_DIR / "research_framed_evaluation.json"
INTERPRETATION_PATH = OUTPUT_DIR / "reviewer_interpretation.md"
INTERNAL_METRICS_PATH = OUTPUT_DIR / "internal_kmeans_metrics.csv"
SURROGATE_DEPTH_PATH = OUTPUT_DIR / "surrogate_depth_sensitivity.csv"
SURROGATE_IMPORTANCE_PATH = OUTPUT_DIR / "surrogate_feature_importance.csv"
STABILITY_PATH = OUTPUT_DIR / "seed_stability_metrics.csv"
CLUSTER_PROFILE_PATH = OUTPUT_DIR / "cluster_profile_summary.csv"
CENTROID_STATS_PATH = OUTPUT_DIR / "centroid_statistics.csv"
DOMINANT_FEATURES_PATH = OUTPUT_DIR / "dominant_behavioral_features.csv"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")


@dataclass(frozen=True)
class EvaluationConfig:
    csv_path: str
    n_clusters: int = 4
    random_state: int = 42
    n_init: int = 20
    k_min: int = 2
    k_max: int = 8
    include_silhouette_reference: bool = True
    surrogate_test_size: float = 0.2
    surrogate_depth_min: int = 1
    surrogate_depth_max: int = 5
    stability_seeds: tuple[int, ...] = (7, 13, 21, 42, 101, 202, 303, 404, 505, 606)
    dominant_feature_top_n: int = 3


@dataclass(frozen=True)
class InternalEvaluationResult:
    selected_k: int
    selected_k_ch_index_model_consistent: float
    selected_k_silhouette_reference_only: float | None
    selected_k_davies_bouldin_index_reference_only: float
    best_k_by_ch_index: int
    selected_k_ch_rank: int
    selected_k_is_best_by_ch_index: bool
    warning: str


@dataclass(frozen=True)
class SurrogateEvaluationResult:
    selected_depth: int
    selected_depth_weighted_f1_surrogate_fidelity: float
    best_depth_by_weighted_f1: int
    best_weighted_f1_surrogate_fidelity: float
    feature_importance: dict[str, float]
    warning: str


@dataclass(frozen=True)
class StabilityEvaluationResult:
    reference_seed: int
    mean_ari_against_reference: float
    min_ari_against_reference: float
    mean_label_agreement_after_centroid_alignment: float
    mean_aligned_centroid_drift: float
    max_aligned_centroid_drift: float
    warning: str


@dataclass(frozen=True)
class ExplainabilityResult:
    cluster_count: int
    feature_count: int
    dominant_feature_top_n: int
    warning: str


@dataclass(frozen=True)
class ResearchFramedEvaluationResult:
    config: EvaluationConfig
    dataset_metadata: dict[str, Any]
    internal_evaluation: InternalEvaluationResult
    surrogate_evaluation: SurrogateEvaluationResult
    stability_evaluation: StabilityEvaluationResult
    explainability: ExplainabilityResult


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _fit_kmeans(X_scaled: np.ndarray, config: EvaluationConfig, seed: int) -> KMeans:
    return KMeans(
        n_clusters=config.n_clusters,
        random_state=seed,
        n_init=config.n_init,
    ).fit(X_scaled)


def _safe_silhouette_score(X_scaled: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None
    return float(silhouette_score(X_scaled, labels))


def _find_min_cost_assignment(cost_matrix: np.ndarray) -> tuple[int, ...]:
    """Return a minimum-cost centroid assignment.

    The returned tuple maps reference centroid index -> candidate centroid index.
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows != n_cols:
        raise ValueError("Centroid alignment requires equal numbers of centroids.")

    # [FIXED] Replace K=4-only permutation search with scipy's Hungarian solver.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ordered_assignment = np.empty(n_rows, dtype=int)
    ordered_assignment[row_ind] = col_ind
    return tuple(int(idx) for idx in ordered_assignment)


def _align_centroids_to_reference(
    reference_centroids: np.ndarray,
    candidate_centroids: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    cost_matrix = np.linalg.norm(
        reference_centroids[:, None, :] - candidate_centroids[None, :, :],
        axis=2,
    )
    ref_to_candidate = _find_min_cost_assignment(cost_matrix)

    aligned = np.empty_like(candidate_centroids)
    candidate_to_ref: dict[int, int] = {}
    for ref_idx, candidate_idx in enumerate(ref_to_candidate):
        aligned[ref_idx] = candidate_centroids[candidate_idx]
        candidate_to_ref[int(candidate_idx)] = int(ref_idx)

    centroid_distances = np.linalg.norm(aligned - reference_centroids, axis=1)
    return aligned, candidate_to_ref, centroid_distances


def _align_labels(labels: np.ndarray, candidate_to_ref: dict[int, int]) -> np.ndarray:
    aligned = np.empty_like(labels)
    for candidate_label, reference_label in candidate_to_ref.items():
        aligned[labels == candidate_label] = reference_label
    return aligned


def load_behavioral_features(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, object, dict[str, Any]]:
    X_derived, _, metadata = load_and_preprocess(csv_path)
    X_scaled, scaler = build_minmax_pipeline(X_derived)
    return X_derived, X_scaled, scaler, metadata


def evaluate_internal_criteria(
    X_scaled: np.ndarray,
    config: EvaluationConfig,
) -> tuple[InternalEvaluationResult, pd.DataFrame]:
    rows: list[dict[str, Any]] = []

    for k in range(config.k_min, config.k_max + 1):
        model = KMeans(n_clusters=k, random_state=config.random_state, n_init=config.n_init)
        labels = model.fit_predict(X_scaled)
        ch_index = float(calinski_harabasz_score(X_scaled, labels))
        # [FIXED] DBI is recorded for reviewer context only; it is not used below.
        davies_bouldin_index_reference_only = float(
            davies_bouldin_score(X_scaled, labels)
        )
        silhouette_reference = (
            _safe_silhouette_score(X_scaled, labels)
            if config.include_silhouette_reference
            else None
        )
        rows.append(
            {
                "k": k,
                "is_selected_persona_k": k == config.n_clusters,
                "calinski_harabasz_index_model_consistent": ch_index,
                "silhouette_score_reference_only": silhouette_reference,
                "davies_bouldin_index_reference_only": (
                    davies_bouldin_index_reference_only
                ),
                "interpretation": (
                    "CH is a K-means-consistent internal criterion; silhouette is "
                    "reported only as a reference because non-convex distributions can "
                    "make it pessimistic; DBI is also reference-only and is not used "
                    "for K selection."
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    selected_row = metrics.loc[metrics["k"] == config.n_clusters].iloc[0]
    best_row = metrics.loc[
        metrics["calinski_harabasz_index_model_consistent"].idxmax()
    ]
    ranked = metrics.sort_values(
        "calinski_harabasz_index_model_consistent",
        ascending=False,
    ).reset_index(drop=True)
    selected_k_ch_rank = int(
        ranked.index[ranked["k"] == config.n_clusters][0] + 1
    )
    selected_k_is_best = int(best_row["k"]) == config.n_clusters

    result = InternalEvaluationResult(
        selected_k=config.n_clusters,
        selected_k_ch_index_model_consistent=float(
            selected_row["calinski_harabasz_index_model_consistent"]
        ),
        selected_k_silhouette_reference_only=(
            None
            if pd.isna(selected_row["silhouette_score_reference_only"])
            else float(selected_row["silhouette_score_reference_only"])
        ),
        selected_k_davies_bouldin_index_reference_only=float(
            selected_row["davies_bouldin_index_reference_only"]
        ),
        best_k_by_ch_index=int(best_row["k"]),
        selected_k_ch_rank=selected_k_ch_rank,
        selected_k_is_best_by_ch_index=selected_k_is_best,
        warning=(
            "Do not present CH Index as universal cluster validity. It is an "
            "internal, variance-ratio criterion aligned with centroid-based K-means."
        ),
    )
    return result, metrics


def evaluate_surrogate_fidelity(
    X_features: pd.DataFrame,
    X_scaled: np.ndarray,
    config: EvaluationConfig,
) -> tuple[SurrogateEvaluationResult, pd.DataFrame, pd.DataFrame, str]:
    # [FIXED] Split before fitting K-means, then create pseudo-labels from a
    # train-only K-means model to avoid evaluating the surrogate on labels from a
    # model fitted with held-out rows.
    row_indices = np.arange(len(X_features))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=config.surrogate_test_size,
        random_state=config.random_state,
    )
    X_train_features = X_features.iloc[train_idx]
    X_test_features = X_features.iloc[test_idx]
    X_train_scaled = X_scaled[train_idx]
    X_test_scaled = X_scaled[test_idx]

    surrogate_kmeans = _fit_kmeans(X_train_scaled, config, config.random_state)
    y_train = surrogate_kmeans.labels_
    y_test = surrogate_kmeans.predict(X_test_scaled)

    depth_rows: list[dict[str, Any]] = []
    fitted_by_depth: dict[int, DecisionTreeClassifier] = {}
    for depth in range(config.surrogate_depth_min, config.surrogate_depth_max + 1):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=config.random_state)
        tree.fit(X_train_features, y_train)
        y_pred = tree.predict(X_test_features)
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        depth_rows.append(
            {
                "max_depth": depth,
                "weighted_f1_surrogate_fidelity": weighted_f1,
                "interpretation": (
                    "This is pseudo-label reproducibility by a shallow proxy model, "
                    "not clustering accuracy against external ground truth. K-means "
                    "pseudo-labels are generated from the training split only."
                ),
            }
        )
        fitted_by_depth[depth] = tree

    depth_metrics = pd.DataFrame(depth_rows)
    selected_depth = 3 if 3 in fitted_by_depth else config.surrogate_depth_max
    selected_tree = fitted_by_depth[selected_depth]
    selected_f1 = float(
        depth_metrics.loc[
            depth_metrics["max_depth"] == selected_depth,
            "weighted_f1_surrogate_fidelity",
        ].iloc[0]
    )
    best_row = depth_metrics.loc[
        depth_metrics["weighted_f1_surrogate_fidelity"].idxmax()
    ]

    importance = pd.DataFrame(
        {
            "feature": list(X_features.columns),
            "decision_tree_importance": selected_tree.feature_importances_,
        }
    ).sort_values("decision_tree_importance", ascending=False)

    feature_importance = {
        str(row.feature): float(row.decision_tree_importance)
        for row in importance.itertuples(index=False)
    }
    tree_rules = export_text(
        selected_tree,
        feature_names=[str(col) for col in X_features.columns],
    )

    result = SurrogateEvaluationResult(
        selected_depth=selected_depth,
        selected_depth_weighted_f1_surrogate_fidelity=selected_f1,
        best_depth_by_weighted_f1=int(best_row["max_depth"]),
        best_weighted_f1_surrogate_fidelity=float(
            best_row["weighted_f1_surrogate_fidelity"]
        ),
        feature_importance=feature_importance,
        warning=(
            "Surrogate F1 measures how well a shallow tree reproduces K-means "
            "assignments. It does not validate the semantic reality of personas."
        ),
    )
    return result, depth_metrics, importance, tree_rules


def evaluate_seed_stability(
    X_scaled: np.ndarray,
    reference_model: KMeans,
    reference_labels: np.ndarray,
    config: EvaluationConfig,
) -> tuple[StabilityEvaluationResult, pd.DataFrame]:
    reference_centroids = reference_model.cluster_centers_
    rows: list[dict[str, Any]] = []

    for seed in config.stability_seeds:
        model = _fit_kmeans(X_scaled, config, seed)
        labels = model.labels_
        aligned_centroids, candidate_to_ref, centroid_distances = (
            _align_centroids_to_reference(reference_centroids, model.cluster_centers_)
        )
        aligned_labels = _align_labels(labels, candidate_to_ref)

        rows.append(
            {
                "seed": seed,
                "adjusted_rand_index_vs_reference": float(
                    adjusted_rand_score(reference_labels, labels)
                ),
                "label_agreement_after_centroid_alignment": float(
                    np.mean(reference_labels == aligned_labels)
                ),
                "mean_aligned_centroid_drift": float(np.mean(centroid_distances)),
                "max_aligned_centroid_drift": float(np.max(centroid_distances)),
                "centroid_alignment_mapping_candidate_to_reference": json.dumps(
                    candidate_to_ref,
                    sort_keys=True,
                ),
                "interpretation": (
                    "ARI captures seed-level assignment consistency; centroid drift "
                    "captures prototype movement after label alignment."
                ),
            }
        )

    metrics = pd.DataFrame(rows)
    # [FIXED] Exclude the reference seed from aggregate means so the perfect
    # self-comparison row cannot inflate stability estimates.
    aggregate_metrics = metrics.loc[metrics["seed"] != config.random_state]
    if aggregate_metrics.empty:
        raise ValueError(
            "stability_seeds must include at least one seed different from random_state."
        )
    result = StabilityEvaluationResult(
        reference_seed=config.random_state,
        mean_ari_against_reference=float(
            aggregate_metrics["adjusted_rand_index_vs_reference"].mean()
        ),
        min_ari_against_reference=float(
            aggregate_metrics["adjusted_rand_index_vs_reference"].min()
        ),
        mean_label_agreement_after_centroid_alignment=float(
            aggregate_metrics["label_agreement_after_centroid_alignment"].mean()
        ),
        mean_aligned_centroid_drift=float(
            aggregate_metrics["mean_aligned_centroid_drift"].mean()
        ),
        max_aligned_centroid_drift=float(
            aggregate_metrics["max_aligned_centroid_drift"].max()
        ),
        warning=(
            "High stability supports operational reproducibility, but it is not "
            "external validation of natural clusters. Aggregate means exclude the "
            "reference seed to avoid artificial inflation."
        ),
    )
    return result, metrics


def build_cluster_profile(
    X_features: pd.DataFrame,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    scaler: object,
    config: EvaluationConfig,
) -> tuple[ExplainabilityResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = X_features.copy()
    labeled["cluster_id"] = labels

    profile_rows: list[dict[str, Any]] = []
    for cluster_id, group in labeled.groupby("cluster_id", sort=True):
        feature_group = group.drop(columns=["cluster_id"])
        row: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "count": int(len(group)),
            "ratio": float(len(group) / len(labeled)),
        }
        for feature in X_features.columns:
            row[f"{feature}__mean"] = float(feature_group[feature].mean())
            row[f"{feature}__median"] = float(feature_group[feature].median())
            row[f"{feature}__std"] = float(feature_group[feature].std(ddof=0))
            row[f"{feature}__min"] = float(feature_group[feature].min())
            row[f"{feature}__max"] = float(feature_group[feature].max())
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows)

    centroids_original = scaler.inverse_transform(model.cluster_centers_)
    centroid_rows: list[dict[str, Any]] = []
    for cluster_id in range(config.n_clusters):
        for feature_idx, feature in enumerate(X_features.columns):
            centroid_rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "centroid_value_original_feature_space": float(
                        centroids_original[cluster_id, feature_idx]
                    ),
                    "centroid_value_scaled_kmeans_space": float(
                        model.cluster_centers_[cluster_id, feature_idx]
                    ),
                }
            )
    centroid_stats = pd.DataFrame(centroid_rows)

    global_mean = X_features.mean(axis=0)
    global_std = X_features.std(axis=0, ddof=0).replace(0.0, np.nan)
    dominant_rows: list[dict[str, Any]] = []
    for cluster_id in range(config.n_clusters):
        centroid_series = pd.Series(centroids_original[cluster_id], index=X_features.columns)
        standardized_delta = ((centroid_series - global_mean) / global_std).fillna(0.0)
        top_features = standardized_delta.abs().sort_values(ascending=False).head(
            config.dominant_feature_top_n
        )
        for rank, feature in enumerate(top_features.index, start=1):
            dominant_rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "feature": feature,
                    "centroid_minus_global_mean": float(
                        centroid_series[feature] - global_mean[feature]
                    ),
                    "standardized_directional_delta": float(
                        standardized_delta[feature]
                    ),
                    "direction": "above_global_mean"
                    if standardized_delta[feature] >= 0
                    else "below_global_mean",
                    "interpretation": (
                        "Dominant features are centroid deviations from the global "
                        "behavioral baseline, not causal explanations."
                    ),
                }
            )
    dominant_features = pd.DataFrame(dominant_rows)

    result = ExplainabilityResult(
        cluster_count=config.n_clusters,
        feature_count=len(X_features.columns),
        dominant_feature_top_n=config.dominant_feature_top_n,
        warning=(
            "Cluster profiles and dominant features support persona interpretation, "
            "but should be reviewed with domain knowledge before naming personas."
        ),
    )
    return result, profile, centroid_stats, dominant_features


def build_reviewer_interpretation(
    result: ResearchFramedEvaluationResult,
    tree_rules: str,
) -> str:
    internal = result.internal_evaluation
    surrogate = result.surrogate_evaluation
    stability = result.stability_evaluation
    if internal.selected_k_is_best_by_ch_index:
        ch_claim_boundary = (
            f"In this run, selected K={internal.selected_k} is also the best K by "
            "the CH sweep. This supports K=4 within the selected K-means criterion, "
            "but still does not prove natural cluster recovery."
        )
    else:
        ch_claim_boundary = (
            f"In this run, selected K={internal.selected_k} ranks "
            f"{internal.selected_k_ch_rank} by CH and the best CH value occurs at "
            f"K={internal.best_k_by_ch_index}. Therefore, CH alone should not be "
            "used to claim K=4 is the internal optimum. K=4 must be defended as an "
            "operational persona granularity supported by stability, surrogate "
            "fidelity, and interpretability."
        )

    return f"""# Reviewer-Style Interpretation

## Framing

This pipeline evaluates K-means as an operational persona assignment mechanism, not as a proof that the data contain spherical or naturally convex clusters. The target claim should be: K=4 gives stable, explainable, and reproducible persona partitions under a centroid-based assignment model.

## Internal Evaluation

Calinski-Harabasz Index is reported as `calinski_harabasz_index_model_consistent`. It is appropriate here because it is aligned with the variance decomposition that K-means optimizes. It should not be described as a universal validity metric for non-convex behavioral data.

Selected K={internal.selected_k} CH Index: {internal.selected_k_ch_index_model_consistent:.6f}

Selected K={internal.selected_k} DBI reference-only value: {internal.selected_k_davies_bouldin_index_reference_only:.6f}

Best K by CH sweep: {internal.best_k_by_ch_index}

{ch_claim_boundary}

Silhouette, if present, is labeled `silhouette_score_reference_only`. DBI is labeled `davies_bouldin_index_reference_only`. Low silhouette or high DBI values do not automatically invalidate the persona system because elongated or L-shaped distributions can be penalized by these geometries.

## Surrogate Fidelity

Decision Tree weighted F1 is reported as `weighted_f1_surrogate_fidelity`. It measures how reproducibly a shallow rule-based model can approximate K-means assignments.

Selected depth={surrogate.selected_depth} weighted F1: {surrogate.selected_depth_weighted_f1_surrogate_fidelity:.6f}

This score does not prove external clustering accuracy, semantic persona correctness, or ground-truth recovery. It is a self-distillation style proxy from K-means pseudo-labels to an interpretable model. The main risk is overinterpreting high F1 as evidence that the clusters are real rather than merely easy to approximate.

## Seed Stability

Random-seed consistency is evaluated using ARI and centroid drift after centroid alignment. ARI checks whether assignments are consistent under different K-means initializations. Centroid drift checks whether persona prototypes move materially.

Mean ARI vs reference: {stability.mean_ari_against_reference:.6f}

Minimum ARI vs reference: {stability.min_ari_against_reference:.6f}

Mean label agreement after centroid alignment: {stability.mean_label_agreement_after_centroid_alignment:.6f}

Mean aligned centroid drift: {stability.mean_aligned_centroid_drift:.6f}

Maximum aligned centroid drift: {stability.max_aligned_centroid_drift:.6f}

High stability supports operational reproducibility. It still does not prove that K-means has recovered natural non-convex clusters.

## K-means Limitation

K-means imposes centroid-based Voronoi partitions and cannot faithfully recover arbitrary non-convex manifolds. On L-shaped smartphone behavior distributions, it should be interpreted as prototype-based behavioral quantization rather than density-faithful cluster discovery.

## Why K-means Is Still Operationally Defensible

K-means remains useful because it gives lightweight inference, stable centroids, simple deployment, and persona prototypes that can be inspected and converted into feedback rules. This is consistent with a behavior-feedback system where stable and explainable assignment is more important than perfect natural cluster recovery.

## Suggested Claim Boundary

Avoid: "K=4 is mathematically proven to be the true number of clusters."

Use: "Under a K-means persona assignment framework, K=4 is evaluated using a model-consistent internal criterion, surrogate fidelity, seed stability, and explainable cluster profiles."

## Selected Surrogate Tree Rules

```text
{tree_rules}
```
"""


def run_research_framed_evaluation(config: EvaluationConfig) -> ResearchFramedEvaluationResult:
    X_features, X_scaled, scaler, metadata = load_behavioral_features(Path(config.csv_path))

    internal_result, internal_metrics = evaluate_internal_criteria(X_scaled, config)

    reference_model = _fit_kmeans(X_scaled, config, config.random_state)
    reference_labels = reference_model.labels_

    surrogate_result, depth_metrics, feature_importance, tree_rules = (
        evaluate_surrogate_fidelity(X_features, X_scaled, config)
    )
    stability_result, stability_metrics = evaluate_seed_stability(
        X_scaled,
        reference_model,
        reference_labels,
        config,
    )
    explainability_result, cluster_profile, centroid_stats, dominant_features = (
        build_cluster_profile(
            X_features,
            X_scaled,
            reference_labels,
            reference_model,
            scaler,
            config,
        )
    )

    result = ResearchFramedEvaluationResult(
        config=config,
        dataset_metadata={
            **metadata,
            "source_csv": str(config.csv_path),
            "n_rows_used_for_evaluation": int(len(X_features)),
            "feature_names": list(map(str, X_features.columns)),
            "scaling": "MinMaxScaler fitted on derived behavioral features",
        },
        internal_evaluation=internal_result,
        surrogate_evaluation=surrogate_result,
        stability_evaluation=stability_result,
        explainability=explainability_result,
    )

    internal_metrics.to_csv(INTERNAL_METRICS_PATH, index=False)
    depth_metrics.to_csv(SURROGATE_DEPTH_PATH, index=False)
    feature_importance.to_csv(SURROGATE_IMPORTANCE_PATH, index=False)
    stability_metrics.to_csv(STABILITY_PATH, index=False)
    cluster_profile.to_csv(CLUSTER_PROFILE_PATH, index=False)
    centroid_stats.to_csv(CENTROID_STATS_PATH, index=False)
    dominant_features.to_csv(DOMINANT_FEATURES_PATH, index=False)

    RESULT_JSON_PATH.write_text(
        json.dumps(_to_builtin(asdict(result)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    INTERPRETATION_PATH.write_text(
        build_reviewer_interpretation(result, tree_rules),
        encoding="utf-8",
    )

    return result


def main() -> None:
    config = EvaluationConfig(csv_path=str(CSV_PATH))
    result = run_research_framed_evaluation(config)

    print(f"Saved structured result: {RESULT_JSON_PATH}")
    print(f"Saved reviewer interpretation: {INTERPRETATION_PATH}")
    print(f"Saved internal metrics: {INTERNAL_METRICS_PATH}")
    print(f"Saved surrogate depth sensitivity: {SURROGATE_DEPTH_PATH}")
    print(f"Saved surrogate feature importance: {SURROGATE_IMPORTANCE_PATH}")
    print(f"Saved stability metrics: {STABILITY_PATH}")
    print(f"Saved cluster profile: {CLUSTER_PROFILE_PATH}")
    print(f"Saved centroid statistics: {CENTROID_STATS_PATH}")
    print(f"Saved dominant features: {DOMINANT_FEATURES_PATH}")
    print(
        "Selected K=4 CH Index "
        f"(model-consistent): {result.internal_evaluation.selected_k_ch_index_model_consistent:.6f}"
    )
    print(
        "Decision Tree weighted F1 "
        f"(surrogate fidelity): {result.surrogate_evaluation.selected_depth_weighted_f1_surrogate_fidelity:.6f}"
    )
    print(
        "Mean ARI across K-means seeds "
        f"(stability): {result.stability_evaluation.mean_ari_against_reference:.6f}"
    )


if __name__ == "__main__":
    main()
