from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_PATH = OUTPUT_DIR / "kmeans_k4_validation.png"
DATA_PATH = OUTPUT_DIR / "l_shaped_sessions.npy"
METRICS_PATH = OUTPUT_DIR / "kmeans_ch_scores.csv"
REPORT_PATH = OUTPUT_DIR / "summary.txt"
RANDOM_STATE = 42

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def _select_font() -> str:
    font_names = {font.name for font in fm.fontManager.ttflist}
    if "Malgun Gothic" in font_names:
        return "Malgun Gothic"
    return "DejaVu Sans"


def generate_l_shaped_sessions(n_samples: int = 5000) -> np.ndarray:
    if n_samples != 5000:
        raise ValueError("This demo is calibrated for exactly 5000 synthetic sessions.")

    rng = np.random.default_rng(RANDOM_STATE)

    specs = [
        ((1.30, 1.20), (0.45, 0.40), 1300),
        ((1.20, 4.00), (0.42, 0.65), 1200),
        ((1.30, 7.70), (0.45, 0.70), 1200),
        ((6.00, 1.20), (0.80, 0.42), 1300),
    ]

    blocks = []
    for mean, std, count in specs:
        x = rng.normal(mean[0], std[0], count)
        y = rng.normal(mean[1], std[1], count)
        blocks.append(np.column_stack([x, y]))

    data = np.clip(np.vstack(blocks), 0.0, 10.0)

    high_high_mask = (data[:, 0] > 5.6) & (data[:, 1] > 3.0)
    if np.any(high_high_mask):
        data[high_high_mask, 1] = rng.normal(1.20, 0.25, high_high_mask.sum())

    return data


def compute_ch_scores(data: np.ndarray) -> tuple[list[int], list[float]]:
    k_values = list(range(2, 9))
    scores: list[float] = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(data)
        scores.append(calinski_harabasz_score(data, labels))

    return k_values, scores


def fit_surrogate_model(
    data: np.ndarray, labels: np.ndarray
) -> tuple[DecisionTreeClassifier, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        data,
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


def save_metrics_csv(k_values: list[int], scores: list[float]) -> None:
    lines = ["k,ch_index"]
    lines.extend(f"{k},{score:.6f}" for k, score in zip(k_values, scores))
    METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_validation_figure(
    data: np.ndarray,
    k_values: list[int],
    scores: list[float],
    labels: np.ndarray,
    centers: np.ndarray,
    tree: DecisionTreeClassifier,
    weighted_f1: float,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rc("font", family=_select_font())
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    palette = sns.color_palette("Set2", 4)
    point_colors = np.array([palette[label] for label in labels])

    x_min, x_max = data[:, 0].min() - 0.4, data[:, 0].max() + 0.4
    y_min, y_max = data[:, 1].min() - 0.4, data[:, 1].max() + 0.4

    best_score = scores[k_values.index(4)]
    ymax = max(scores) * 1.08
    ymin = min(scores) * 0.92
    axes[0].plot(
        k_values,
        scores,
        color="#2F6BFF",
        linewidth=3,
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    axes[0].axvline(4, color="red", linestyle="--", linewidth=1.6, alpha=0.8)
    axes[0].scatter([4], [best_score], color="red", s=130, zorder=4)
    axes[0].annotate(
        f"K=4\nCH={best_score:,.0f}",
        xy=(4, best_score),
        xytext=(4.35, best_score - (ymax - ymin) * 0.08),
        fontsize=11,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.5},
    )
    axes[0].set_title("CH Index 변화", fontsize=16, weight="bold")
    axes[0].set_xlabel("군집 수 K", fontsize=12)
    axes[0].set_ylabel("Calinski-Harabasz Index", fontsize=12)
    axes[0].set_xticks(k_values)
    axes[0].set_ylim(ymin, ymax)

    axes[1].scatter(
        data[:, 0],
        data[:, 1],
        s=15,
        c=point_colors,
        alpha=0.75,
        linewidths=0,
    )
    axes[1].scatter(
        centers[:, 0],
        centers[:, 1],
        s=260,
        c="black",
        marker="X",
        linewidths=1.4,
        edgecolors="white",
        label="중심점",
    )
    axes[1].set_title("K=4 K-Means 산점도", fontsize=16, weight="bold")
    axes[1].set_xlabel("집중도", fontsize=12)
    axes[1].set_ylabel("전환율", fontsize=12)
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].legend(loc="upper right")
    axes[1].text(
        0.03,
        0.05,
        "우상단 High-High 영역 비움",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    z = tree.predict(grid).reshape(xx.shape)

    axes[2].contourf(
        xx,
        yy,
        z,
        levels=np.arange(5) - 0.5,
        cmap="Pastel2",
        alpha=0.75,
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
        data[:, 0],
        data[:, 1],
        s=12,
        c=point_colors,
        alpha=0.45,
        linewidths=0,
    )
    axes[2].set_title(
        f"의사결정나무 결정 경계\nTest F1-Score: {weighted_f1 * 100:.2f}%",
        fontsize=16,
        weight="bold",
    )
    axes[2].set_xlabel("집중도", fontsize=12)
    axes[2].set_ylabel("전환율", fontsize=12)
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)

    fig.suptitle(
        "L자형 스마트폰 세션 데이터에서 K=4의 유의성 검증",
        fontsize=18,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = generate_l_shaped_sessions()
    np.save(DATA_PATH, data)

    k_values, scores = compute_ch_scores(data)
    save_metrics_csv(k_values, scores)

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    tree, weighted_f1 = fit_surrogate_model(data, labels)

    plot_validation_figure(data, k_values, scores, labels, centers, tree, weighted_f1)

    REPORT_PATH.write_text(
        "\n".join(
            [
                f"figure_path={FIGURE_PATH.name}",
                f"data_path={DATA_PATH.name}",
                f"metrics_path={METRICS_PATH.name}",
                f"k4_ch_index={scores[k_values.index(4)]:.6f}",
                f"weighted_f1={weighted_f1:.6f}",
                f"best_k_by_ch={k_values[int(np.argmax(scores))]}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved figure to: {FIGURE_PATH}")
    print(f"Saved data to: {DATA_PATH}")
    print(f"Saved CH metrics to: {METRICS_PATH}")
    print(f"Saved summary to: {REPORT_PATH}")
    print(f"K=4 CH Index: {scores[k_values.index(4)]:.2f}")
    print(f"Decision tree weighted F1: {weighted_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
