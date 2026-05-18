from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = PROJECT_ROOT / "clustering_pipeline"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from clustering_pipeline.preprocessing import build_minmax_pipeline, load_and_preprocess


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "figure"

SUMMARY_PATH = BASE_DIR / "real_data_summary.txt"
INTERNAL_METRICS_PATH = BASE_DIR / "internal_kmeans_metrics.csv"
SEED_STABILITY_PATH = BASE_DIR / "seed_stability_metrics.csv"
SURROGATE_DEPTH_PATH = BASE_DIR / "surrogate_depth_sensitivity.csv"

SCATTER_PATH = OUTPUT_DIR / "behavior_scatter_centroids.png"
CH_PATH = OUTPUT_DIR / "k_vs_ch_index.png"
SEED_STABILITY_PATH_OUT = OUTPUT_DIR / "seed_stability.png"
SURROGATE_DEPTH_PATH_OUT = OUTPUT_DIR / "tree_depth_vs_surrogate_f1.png"


def _parse_summary_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return None


def _resolve_source_csv() -> Path:
    source_csv = _parse_summary_value(SUMMARY_PATH, "source_csv")
    if not source_csv:
        raise FileNotFoundError(
            f"Missing source_csv in summary file: {SUMMARY_PATH}"
        )

    candidate = PROJECT_ROOT / source_csv
    if candidate.exists():
        return candidate

    fallback = PROJECT_ROOT / "past_data" / source_csv
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find source CSV: {source_csv} under {PROJECT_ROOT}"
    )


def generate_scatter_with_centroids() -> None:
    csv_path = _resolve_source_csv()
    X_derived, _, _ = load_and_preprocess(csv_path)
    X_scaled, scaler = build_minmax_pipeline(X_derived)

    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.scatter(
        X_derived["focus_intensity"],
        X_derived["switch_frequency"],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.6,
        linewidths=0,
    )
    ax.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=240,
        c="black",
        marker="X",
        edgecolors="white",
        linewidths=1.2,
        label="centroid",
    )
    ax.set_title("Behavior Space Scatter + Centroids", fontsize=14)
    ax.set_xlabel("focus_intensity")
    ax.set_ylabel("switch_frequency")
    ax.legend(loc="upper right")
    fig.text(
        0.01,
        0.01,
        "Derived features; K=4 KMeans fit on MinMax-scaled data.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=180)
    plt.close(fig)


def generate_k_vs_ch() -> None:
    df = pd.read_csv(INTERNAL_METRICS_PATH)
    k_values = df["k"].to_numpy()
    ch_values = df["calinski_harabasz_index_model_consistent"].to_numpy()
    selected_mask = df["is_selected_persona_k"]
    selected_k = int(df.loc[selected_mask, "k"].iloc[0])
    selected_ch = float(
        df.loc[selected_mask, "calinski_harabasz_index_model_consistent"].iloc[0]
    )
    best_k = int(df.loc[df["calinski_harabasz_index_model_consistent"].idxmax(), "k"])

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(k_values, ch_values, marker="o", linewidth=2.5, color="#2F6BFF")
    ax.scatter([selected_k], [selected_ch], color="red", s=90, zorder=3)
    ax.axvline(selected_k, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("K vs CH Index", fontsize=14)
    ax.set_xlabel("K")
    ax.set_ylabel("Calinski-Harabasz Index (model-consistent)")
    ax.set_xticks(k_values)
    fig.text(
        0.01,
        0.01,
        f"Selected persona K={selected_k}; best CH at K={best_k}.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(CH_PATH, dpi=180)
    plt.close(fig)


def generate_seed_stability() -> None:
    df = pd.read_csv(SEED_STABILITY_PATH)
    seeds = df["seed"].to_numpy()
    ari = df["adjusted_rand_index_vs_reference"].to_numpy()
    mean_drift = df["mean_aligned_centroid_drift"].to_numpy()
    max_drift = df["max_aligned_centroid_drift"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True)
    axes[0].plot(seeds, ari, marker="o", linewidth=2.2, color="#2F6BFF")
    axes[0].set_title("Seed Stability: ARI", fontsize=13)
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel("Adjusted Rand Index vs reference")

    axes[1].plot(seeds, mean_drift, marker="o", linewidth=2.2, color="#FF7A45")
    axes[1].plot(seeds, max_drift, marker="o", linewidth=1.4, linestyle="--", color="#7A7A7A")
    axes[1].set_title("Seed Stability: Centroid Drift", fontsize=13)
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("Aligned centroid drift (scaled space)")
    axes[1].legend(["mean drift", "max drift"], loc="upper right")

    fig.text(
        0.01,
        0.01,
        "Centroid drift is computed after alignment in MinMax-scaled feature space.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(SEED_STABILITY_PATH_OUT, dpi=180)
    plt.close(fig)


def generate_surrogate_depth() -> None:
    df = pd.read_csv(SURROGATE_DEPTH_PATH)
    depths = df["max_depth"].to_numpy()
    f1_scores = df["weighted_f1_surrogate_fidelity"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(depths, f1_scores, marker="o", linewidth=2.5, color="#2F6BFF")
    ax.set_title("Tree Depth vs Surrogate F1", fontsize=14)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Weighted F1 (surrogate fidelity)")
    ax.set_xticks(depths)
    fig.text(
        0.01,
        0.01,
        "F1 reflects pseudo-label reproducibility, not external clustering accuracy.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(SURROGATE_DEPTH_PATH_OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_scatter_with_centroids()
    generate_k_vs_ch()
    generate_seed_stability()
    generate_surrogate_depth()


if __name__ == "__main__":
    main()
