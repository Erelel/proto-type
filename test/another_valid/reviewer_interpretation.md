# Reviewer-Style Interpretation

## Framing

This pipeline evaluates K-means as an operational persona assignment mechanism, not as a proof that the data contain spherical or naturally convex clusters. The target claim should be: K=4 gives stable, explainable, and reproducible persona partitions under a centroid-based assignment model.

## Internal Evaluation

Calinski-Harabasz Index is reported as `calinski_harabasz_index_model_consistent`. It is appropriate here because it is aligned with the variance decomposition that K-means optimizes. It should not be described as a universal validity metric for non-convex behavioral data.

Selected K=4 CH Index: 11421.358901

Selected K=4 DBI reference-only value: 0.703708

Best K by CH sweep: 8

In this run, selected K=4 ranks 5 by CH and the best CH value occurs at K=8. Therefore, CH alone should not be used to claim K=4 is the internal optimum. K=4 must be defended as an operational persona granularity supported by stability, surrogate fidelity, and interpretability.

Silhouette, if present, is labeled `silhouette_score_reference_only`. DBI is labeled `davies_bouldin_index_reference_only`. Low silhouette or high DBI values do not automatically invalidate the persona system because elongated or L-shaped distributions can be penalized by these geometries.

## Surrogate Fidelity

Decision Tree weighted F1 is reported as `weighted_f1_surrogate_fidelity`. It measures how reproducibly a shallow rule-based model can approximate K-means assignments.

Selected depth=3 weighted F1: 0.963132

This score does not prove external clustering accuracy, semantic persona correctness, or ground-truth recovery. It is a self-distillation style proxy from K-means pseudo-labels to an interpretable model. The main risk is overinterpreting high F1 as evidence that the clusters are real rather than merely easy to approximate.

## Seed Stability

Random-seed consistency is evaluated using ARI and centroid drift after centroid alignment. ARI checks whether assignments are consistent under different K-means initializations. Centroid drift checks whether persona prototypes move materially.

Mean ARI vs reference: 0.992141

Minimum ARI vs reference: 0.983445

Mean label agreement after centroid alignment: 0.997119

Mean aligned centroid drift: 0.001204

Maximum aligned centroid drift: 0.004447

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
|--- focus_intensity <= 4.03
|   |--- switch_frequency <= 0.60
|   |   |--- focus_intensity <= 3.97
|   |   |   |--- class: 2
|   |   |--- focus_intensity >  3.97
|   |   |   |--- class: 2
|   |--- switch_frequency >  0.60
|   |   |--- switch_frequency <= 1.96
|   |   |   |--- class: 3
|   |   |--- switch_frequency >  1.96
|   |   |   |--- class: 1
|--- focus_intensity >  4.03
|   |--- switch_frequency <= 0.86
|   |   |--- switch_frequency <= 0.66
|   |   |   |--- class: 0
|   |   |--- switch_frequency >  0.66
|   |   |   |--- class: 0
|   |--- switch_frequency >  0.86
|   |   |--- focus_intensity <= 4.98
|   |   |   |--- class: 3
|   |   |--- focus_intensity >  4.98
|   |   |   |--- class: 0

```
