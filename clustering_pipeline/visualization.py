import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from constants import SKEWNESS_THRESHOLD


def _bias_label(skew_value: float) -> str:
    if skew_value > SKEWNESS_THRESHOLD:
        return "right-skewed"
    if skew_value < -SKEWNESS_THRESHOLD:
        return "left-skewed"
    return "near-balanced"


def plot_source_histograms(
    source_df: pd.DataFrame,
    output_path: Path,
    skewness: dict[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "foreground_app_duration_sum",
        "foreground_app_switch_per_hour",
    ]
    titles = [
        "Source for force_intensity",
        "Source for switch_frequency",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, column, title in zip(axes, columns, titles):
        values = source_df[column].to_numpy(dtype=float)
        skew_value = float(skewness[column])

        ax.hist(values, bins=60, color="#2A6F97", alpha=0.85, edgecolor="white")
        ax.axvline(values.mean(), color="#D62828", linestyle="--", linewidth=1.5)
        ax.axvline(np.median(values), color="#1B4332", linestyle=":", linewidth=1.5)

        ax.set_title(
            f"{title}\nSkew={skew_value:.2f} ({_bias_label(skew_value)})",
            fontsize=11,
        )
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    fig.suptitle("Histogram-based skewness check for source variables", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_kmeans_clusters_on_derived_features(
    X_derived: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    centroids: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"force_intensity", "switch_frequency"}
    if not required_columns.issubset(X_derived.columns):
        raise ValueError(
            "X_derived must include force_intensity and switch_frequency columns."
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(
        X_derived["force_intensity"],
        X_derived["switch_frequency"],
        c=labels,
        cmap="tab10",
        alpha=0.45,
        s=16,
        linewidths=0,
    )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=130,
            c="black",
            edgecolors="white",
            linewidths=1.0,
            label="Centroids",
        )
        ax.legend(loc="upper right")

    ax.set_title("KMeans clusters on derived features")
    ax.set_xlabel("force_intensity")
    ax.set_ylabel("switch_frequency")
    fig.colorbar(scatter, ax=ax, label="Cluster ID")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_minmax_vs_standard_kmeans_compare(
    X_derived: pd.DataFrame,
    labels_minmax: np.ndarray,
    labels_standard: np.ndarray,
    output_path: Path,
    centroids_minmax: np.ndarray | None = None,
    centroids_standard: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"force_intensity", "switch_frequency"}
    if not required_columns.issubset(X_derived.columns):
        raise ValueError(
            "X_derived must include force_intensity and switch_frequency columns."
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    x = X_derived["force_intensity"]
    y = X_derived["switch_frequency"]

    scatter_minmax = axes[0].scatter(
        x,
        y,
        c=labels_minmax,
        cmap="tab10",
        alpha=0.45,
        s=16,
        linewidths=0,
    )
    axes[0].set_title("MinMaxScaler + KMeans")
    axes[0].set_xlabel("force_intensity")
    axes[0].set_ylabel("switch_frequency")
    fig.colorbar(scatter_minmax, ax=axes[0], label="Cluster ID")

    if centroids_minmax is not None:
        axes[0].scatter(
            centroids_minmax[:, 0],
            centroids_minmax[:, 1],
            marker="X",
            s=260,
            c="black",
            edgecolors="white",
            linewidths=1.0,
        )

    scatter_standard = axes[1].scatter(
        x,
        y,
        c=labels_standard,
        cmap="tab10",
        alpha=0.45,
        s=16,
        linewidths=0,
    )
    axes[1].set_title("StandardScaler + KMeans")
    axes[1].set_xlabel("force_intensity")
    fig.colorbar(scatter_standard, ax=axes[1], label="Cluster ID")

    if centroids_standard is not None:
        axes[1].scatter(
            centroids_standard[:, 0],
            centroids_standard[:, 1],
            marker="X",
            s=260,
            c="black",
            edgecolors="white",
            linewidths=1.0,
        )

    fig.suptitle("minmax_vs_standard_kmeans_compare", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)