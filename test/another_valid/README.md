# Research-Framed Persona Clustering Evaluation

This directory contains validation code for a behavior-based smartphone persona assignment pipeline. The goal is not to prove that the data contain perfectly natural spherical clusters. The goal is to evaluate whether a fixed `K=4` K-means persona assignment is operationally stable, reproducible, and explainable enough for downstream feedback generation.

## Research Framing

- `CH Index` is treated as a K-means-consistent internal criterion, not a universal cluster validity metric.
- `Silhouette Score` is optional and reference-only because elongated or L-shaped behavioral distributions can make it pessimistic.
- `Decision Tree weighted F1` is interpreted as surrogate fidelity and assignment reproducibility, not clustering accuracy.
- `ARI` and aligned centroid drift are used to evaluate random-seed stability.
- Cluster profiles, centroid statistics, and dominant feature deviations support explainability, but persona names should still be reviewed with domain knowledge.

## Main Script

```bash
python test/another_valid/real_data_validation.py
```

The script uses `merged.csv`, the existing preprocessing pipeline, fixed `K=4`, MinMax scaling, K-means, a shallow Decision Tree surrogate, and seed-stability analysis.

## Outputs

- `research_framed_evaluation.json`: structured summary for reports or downstream tooling
- `reviewer_interpretation.md`: automatically generated reviewer-style interpretation layer
- `internal_kmeans_metrics.csv`: CH Index and optional reference-only Silhouette for `K=2..8`
- `surrogate_depth_sensitivity.csv`: weighted F1 for Decision Tree depths `1..5`
- `surrogate_feature_importance.csv`: selected depth-3 tree feature importances
- `seed_stability_metrics.csv`: ARI and aligned centroid drift across random seeds
- `cluster_profile_summary.csv`: per-cluster feature distribution summary
- `centroid_statistics.csv`: centroid values in original feature space and scaled K-means space
- `dominant_behavioral_features.csv`: cluster-level dominant feature deviations from the global mean

## Claim Boundary

Avoid claiming that `K=4` is mathematically proven as the true number of natural clusters. The defensible claim is narrower and stronger:

> Under a K-means persona assignment framework, `K=4` is evaluated using a model-consistent internal criterion, surrogate fidelity, seed stability, and explainable cluster profiles.
