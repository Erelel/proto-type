from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "foreground_app_duration_sum",
    "foreground_app_switch_per_hour",
]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_log1p_domain(series: pd.Series, column_name: str) -> None:
    min_value = float(series.min())
    if min_value < 0:
        raise ValueError(
            f"Column '{column_name}' has min value {min_value}, which is below 0. "
            "Cannot apply log1p directly."
        )


def _plot_hist(ax: plt.Axes, values: pd.Series, title: str, xlabel: str) -> None:
    arr = values.to_numpy(dtype=float)
    ax.hist(arr, bins=60, color="#2A6F97", alpha=0.85, edgecolor="white")
    ax.axvline(arr.mean(), color="#D62828", linestyle="--", linewidth=1.2, label="mean")
    ax.axvline(np.median(arr), color="#1B4332", linestyle=":", linewidth=1.2, label="median")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")


def create_figure(csv_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    duration_raw = df["foreground_app_duration_sum"].dropna()
    switch_raw = df["foreground_app_switch_per_hour"].dropna()

    _validate_log1p_domain(duration_raw, "foreground_app_duration_sum")
    _validate_log1p_domain(switch_raw, "foreground_app_switch_per_hour")

    duration_log = np.log1p(duration_raw)
    switch_log = np.log1p(switch_raw)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_hist(
        axes[0, 0],
        duration_raw,
        f"Raw foreground_app_duration_sum (skew={duration_raw.skew():.2f})",
        "foreground_app_duration_sum",
    )
    _plot_hist(
        axes[0, 1],
        pd.Series(duration_log),
        f"Log1p foreground_app_duration_sum (skew={pd.Series(duration_log).skew():.2f})",
        "log1p(foreground_app_duration_sum)",
    )
    _plot_hist(
        axes[1, 0],
        switch_raw,
        f"Raw foreground_app_switch_per_hour (skew={switch_raw.skew():.2f})",
        "foreground_app_switch_per_hour",
    )
    _plot_hist(
        axes[1, 1],
        pd.Series(switch_log),
        f"Log1p foreground_app_switch_per_hour (skew={pd.Series(switch_log).skew():.2f})",
        "log1p(foreground_app_switch_per_hour)",
    )

    fig.suptitle("Raw vs Log1p Histogram Comparison (2x2)", fontsize=15)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 2x2 histogram figure for raw vs log1p distributions."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mendeley_v4.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result") / "0402_image" / "raw_vs_log1p_hist_2x2.png",
        help="Output image path",
    )
    args = parser.parse_args()

    output = create_figure(args.csv, args.out)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
