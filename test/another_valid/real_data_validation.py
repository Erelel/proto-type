from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / "clustering_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from preprocessing import build_minmax_pipeline, load_and_preprocess


OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "merged.csv"
FIGURE_PATH = OUTPUT_DIR / "real_data_validation.png"
CH_CSV_PATH = OUTPUT_DIR / "real_data_ch_scores.csv"
SUMMARY_PATH = OUTPUT_DIR / "real_data_summary.txt"
RANDOM_STATE = 42

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def _select_font() -> str:
    font_names = {font.name for font in fm.fontManager.ttflist}
    if "Malgun Gothic" in font_names:
        return "Malgun Gothic"
    return "DejaVu Sans"


def load_real_features() -> tuple[pd.DataFrame, np.ndarray, object, dict]:
    X_derived, _, metadata = load_and_preprocess(CSV_PATH)
    X_scaled, scaler = build_minmax_pipeline(X_derived)
    return X_derived, X_scaled, scaler, metadata


def compute_ch_scores(X_scaled: np.ndarray) -> tuple[list[int], list[float]]:
    k_values = list(range(2, 9))
    scores: list[float] = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        scores.append(calinski_harabasz_score(X_scaled, labels))

    return k_values, scores


def fit_k4_model(
    X_scaled: np.ndarray,
    scaler,
) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(X_scaled)
    centroids_raw = scaler.inverse_transform(model.cluster_centers_)
    return labels, centroids_raw


def fit_surrogate_model(
    X_derived: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[DecisionTreeClassifier, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_derived,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    return tree, weighted_f1


def save_ch_scores(k_values: list[int], scores: list[float]) -> None:
    lines = ["k,ch_index"]
    lines.extend(f"{k},{score:.6f}" for k, score in zip(k_values, scores))
    CH_CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_validation_figure(
    X_derived: pd.DataFrame,
    k_values: list[int],
    scores: list[float],
    labels: np.ndarray,
    centroids_raw: np.ndarray,
    tree: DecisionTreeClassifier,
    weighted_f1: float,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rc("font", family=_select_font())
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    palette = sns.color_palette("tab10", 4)
    point_colors = np.array([palette[label] for label in labels])

    x = X_derived["focus_intensity"].to_numpy(dtype=float)
    y = X_derived["switch_frequency"].to_numpy(dtype=float)
    x_min, x_max = x.min() - 0.25, x.max() + 0.25
    y_min, y_max = y.min() - 0.15, y.max() + 0.15

    k4_score = scores[k_values.index(4)]
    best_k = k_values[int(np.argmax(scores))]
    axes[0].plot(
        k_values,
        scores,
        color="#2F6BFF",
        linewidth=3,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    axes[0].scatter([4], [k4_score], color="red", s=120, zorder=4)
    axes[0].annotate(
        f"K=4\nCH={k4_score:,.0f}",
        xy=(4, k4_score),
        xytext=(4.35, k4_score + (max(scores) - min(scores)) * 0.06),
        fontsize=11,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5},
    )
    axes[0].text(
        0.03,
        0.05,
        f"CH 최대값은 K={best_k}",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    axes[0].set_title("실제 데이터 CH Index 변화", fontsize=16, weight="bold")
    axes[0].set_xlabel("군집 수 K", fontsize=12)
    axes[0].set_ylabel("Calinski-Harabasz Index", fontsize=12)
    axes[0].set_xticks(k_values)

    axes[1].scatter(
        x,
        y,
        s=14,
        c=point_colors,
        alpha=0.35,
        linewidths=0,
    )
    axes[1].scatter(
        centroids_raw[:, 0],
        centroids_raw[:, 1],
        s=240,
        c="black",
        marker="X",
        linewidths=1.3,
        edgecolors="white",
        label="중심점",
    )
    axes[1].set_title("실제 데이터 K=4 군집 결과", fontsize=16, weight="bold")
    axes[1].set_xlabel("집중도", fontsize=12)
    axes[1].set_ylabel("전환율", fontsize=12)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].legend(loc="upper right")

    cluster_counts = pd.Series(labels).value_counts().sort_index()
    summary_text = "\n".join(
        f"{cluster_id}: n={int(count)} ({count / len(labels):.1%})"
        for cluster_id, count in cluster_counts.items()
    )
    axes[1].text(
        0.98,
        0.90,
        summary_text,
        transform=axes[1].transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid = pd.DataFrame(
        {
            "focus_intensity": xx.ravel(),
            "switch_frequency": yy.ravel(),
        }
    )
    z = tree.predict(grid).reshape(xx.shape)

    axes[2].contourf(
        xx,
        yy,
        z,
        levels=np.arange(5) - 0.5,
        cmap="Pastel2",
        alpha=0.7,
    )
    axes[2].contour(
        xx,
        yy,
        z,
        levels=np.arange(4),
        colors="#666666",
        linewidths=1.0,
        alpha=0.8,
    )
    axes[2].scatter(
        x,
        y,
        s=12,
        c=point_colors,
        alpha=0.30,
        linewidths=0,
    )
    axes[2].set_title(
        f"의사결정나무 결정 경계\nTest F1-Score: {weighted_f1:.4f}",
        fontsize=16,
        weight="bold",
    )
    axes[2].set_xlabel("집중도", fontsize=12)
    axes[2].set_ylabel("전환율", fontsize=12)
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)

    fig.suptitle(
        "실제 merged.csv 데이터에서 K=4의 서비스적 타당성 검증",
        fontsize=18,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    X_derived, X_scaled, scaler, metadata = load_real_features()
    k_values, scores = compute_ch_scores(X_scaled)
    save_ch_scores(k_values, scores)

    labels, centroids_raw = fit_k4_model(X_scaled, scaler)
    tree, weighted_f1 = fit_surrogate_model(X_derived, labels)

    plot_validation_figure(
        X_derived=X_derived,
        k_values=k_values,
        scores=scores,
        labels=labels,
        centroids_raw=centroids_raw,
        tree=tree,
        weighted_f1=weighted_f1,
    )

    best_k = k_values[int(np.argmax(scores))]
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                f"figure_path={FIGURE_PATH.name}",
                f"ch_csv_path={CH_CSV_PATH.name}",
                f"source_csv={CSV_PATH.name}",
                f"n_rows={len(X_derived)}",
                f"log_transform_applied={metadata['log_transform_applied']}",
                f"k4_ch_index={scores[k_values.index(4)]:.6f}",
                f"best_k_by_ch={best_k}",
                f"weighted_f1={weighted_f1:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved figure to: {FIGURE_PATH}")
    print(f"Saved CH scores to: {CH_CSV_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print(f"K=4 CH Index: {scores[k_values.index(4)]:.4f}")
    print(f"Best K by CH: {best_k}")
    print(f"Decision tree weighted F1: {weighted_f1:.4f}")


if __name__ == "__main__":
    main()
