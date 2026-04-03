import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from constants import MINMAX_FEATURE_RANGE


def _validate_derived_columns(X_derived: pd.DataFrame) -> None:
    required_columns = {"force_intensity", "switch_frequency"}
    if not required_columns.issubset(X_derived.columns):
        raise ValueError(
            "X_derived must include force_intensity and switch_frequency columns."
        )


def plot_derived_feature_histograms(
    X_derived: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_derived_columns(X_derived)

    columns = [
        "force_intensity",
        "switch_frequency",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, column in zip(axes, columns):
        values = X_derived[column].to_numpy(dtype=float)

        ax.hist(values, bins=60, color="#2A6F97", alpha=0.85, edgecolor="white")
        ax.axvline(values.mean(), color="#D62828", linestyle="--", linewidth=1.5)
        ax.axvline(np.median(values), color="#1B4332", linestyle=":", linewidth=1.5)

        ax.set_title(f"{column} (raw derived)", fontsize=11)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    fig.suptitle("Derived Feature Histograms", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_minmax_scaled_derived_histograms(
    X_derived: pd.DataFrame,
    X_minmax_scaled: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_derived_columns(X_derived)

    if X_minmax_scaled.shape != X_derived.shape:
        raise ValueError("X_minmax_scaled must have the same shape as X_derived.")

    scaled_df = pd.DataFrame(
        X_minmax_scaled,
        columns=X_derived.columns,
        index=X_derived.index,
    )

    columns = ["force_intensity", "switch_frequency"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, column in enumerate(columns):
        raw_values = X_derived[column].to_numpy(dtype=float)
        scaled_values = scaled_df[column].to_numpy(dtype=float)

        axes[0, col_idx].hist(
            raw_values, bins=60, color="#2A6F97", alpha=0.85, edgecolor="white"
        )
        axes[0, col_idx].set_title(f"{column} (raw)")
        axes[0, col_idx].set_xlabel(column)
        axes[0, col_idx].set_ylabel("Count")

        axes[1, col_idx].hist(
            scaled_values,
            bins=60,
            color="#F4A261",
            alpha=0.85,
            edgecolor="white",
        )
        axes[1, col_idx].set_title(
            f"{column} (MinMax scaled: {MINMAX_FEATURE_RANGE[0]:.0f} to {MINMAX_FEATURE_RANGE[1]:.0f})"
        )
        axes[1, col_idx].set_xlabel(column)
        axes[1, col_idx].set_ylabel("Count")
        axes[1, col_idx].set_xlim(
            MINMAX_FEATURE_RANGE[0] - 0.02,
            MINMAX_FEATURE_RANGE[1] + 0.02,
        )

    fig.suptitle("Derived Features: Raw vs MinMax Histogram Change", fontsize=13)
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
    _validate_derived_columns(X_derived)

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
