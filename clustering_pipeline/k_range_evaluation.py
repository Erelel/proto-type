from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from constants import (
    DEFAULT_CSV_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MAIN_CSV,
    resolve_project_path,
)
from modeling import fit_cluster_model
from preprocessing import build_minmax_pipeline, load_and_preprocess


def evaluate_k_range(csv_path: Path, k_min: int, k_max: int) -> tuple[pd.DataFrame, dict]:
    if k_min < 2:
        raise ValueError("k_min must be at least 2.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")

    X, _, metadata = load_and_preprocess(csv_path)
    X_scaled, _ = build_minmax_pipeline(X)

    rows: list[dict] = []
    for k in range(k_min, k_max + 1):
        result = fit_cluster_model(X_scaled, "kmeans", k)
        rows.append(
            {
                "k": k,
                "silhouette": float(result["silhouette"]),
                "davies_bouldin": float(result["davies_bouldin"]),
                "inertia": float(result["model"].inertia_),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return metrics_df, metadata


def plot_elbow_like_diagnostics(metrics_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#1D3557")
    axes[0].set_title("Elbow-style Inertia Curve")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (SSE)")
    axes[0].set_xticks(metrics_df["k"].tolist())
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        metrics_df["k"],
        metrics_df["silhouette"],
        marker="o",
        color="#2A9D8F",
        label="Silhouette (higher is better)",
    )
    axes[1].plot(
        metrics_df["k"],
        metrics_df["davies_bouldin"],
        marker="s",
        color="#E76F51",
        label="Davies-Bouldin (lower is better)",
    )
    axes[1].set_title("Cluster Quality Metrics by k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(metrics_df["k"].tolist())
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KMeans for a range of k and save metrics/elbow-like plots."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_MAIN_CSV,
        help="Input CSV path",
    )
    parser.add_argument("--k-min", type=int, default=3, help="Minimum k value")
    parser.add_argument("--k-max", type=int, default=8, help="Maximum k value")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory for CSV output",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory for plot output",
    )
    args = parser.parse_args()

    csv_path = resolve_project_path(args.csv)
    csv_dir = resolve_project_path(args.csv_dir)
    image_dir = resolve_project_path(args.image_dir)

    metrics_df, metadata = evaluate_k_range(csv_path, args.k_min, args.k_max)

    csv_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_output = csv_dir / f"kmeans_k_{args.k_min}_{args.k_max}_metrics.csv"
    image_output = image_dir / f"kmeans_elbow_like_k_{args.k_min}_{args.k_max}.png"

    metrics_df.to_csv(csv_output, index=False)
    plot_elbow_like_diagnostics(metrics_df, image_output)

    best_silhouette_row = metrics_df.loc[metrics_df["silhouette"].idxmax()]
    best_dbi_row = metrics_df.loc[metrics_df["davies_bouldin"].idxmin()]

    print("=== Preprocessing Summary ===")
    print(f"Rows after dropna: {metadata['final_rows']}")
    print(f"Derived features: {metadata['used_columns']}")
    print(f"Log transform applied: {metadata['log_transform_applied']}")

    print("\n=== K-range Metrics ===")
    print(metrics_df.to_string(index=False))

    print("\n=== Best k by metric ===")
    print(
        "Best silhouette -> "
        f"k={int(best_silhouette_row['k'])}, "
        f"score={best_silhouette_row['silhouette']:.6f}"
    )
    print(
        "Best DBI -> "
        f"k={int(best_dbi_row['k'])}, "
        f"score={best_dbi_row['davies_bouldin']:.6f}"
    )

    print("\n=== Saved outputs ===")
    print(f"CSV: {csv_output}")
    print(f"Plot: {image_output}")


if __name__ == "__main__":
    main()
