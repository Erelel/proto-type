from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DURATION_COLUMN = "foreground_app_duration_sum"
SWITCH_COLUMN_CANDIDATES = (
    "switch_per_hour",
    "foreground_app_switch_per_hour",
)


def _resolve_switch_column(df: pd.DataFrame) -> str:
    for column in SWITCH_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Missing switch column. Expected one of: "
        f"{list(SWITCH_COLUMN_CANDIDATES)}"
    )


def _to_numeric_non_null(series: pd.Series, column_name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Column '{column_name}' has no numeric values.")
    return values


def create_histograms(csv_path: Path, output_path: Path, bins: int) -> Path:
    df = pd.read_csv(csv_path)

    if DURATION_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DURATION_COLUMN}")

    switch_column = _resolve_switch_column(df)

    duration_values = _to_numeric_non_null(df[DURATION_COLUMN], DURATION_COLUMN)
    switch_values = _to_numeric_non_null(df[switch_column], switch_column)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        duration_values.to_numpy(),
        bins=bins,
        color="#2A6F97",
        alpha=0.9,
        edgecolor="white",
    )
    axes[0].set_title(DURATION_COLUMN)
    axes[0].set_xlabel(DURATION_COLUMN)
    axes[0].set_ylabel("Count")

    axes[1].hist(
        switch_values.to_numpy(),
        bins=bins,
        color="#F4A261",
        alpha=0.9,
        edgecolor="white",
    )
    axes[1].set_title(switch_column)
    axes[1].set_xlabel(switch_column)
    axes[1].set_ylabel("Count")

    fig.suptitle("Column Histograms")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create pure histograms for selected columns from mendeley_v5.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mendeley_v5.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result") / "column_hist.png",
        help="Output image path",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins",
    )
    args = parser.parse_args()

    saved_path = create_histograms(args.csv, args.out, args.bins)
    print(f"Saved histogram image: {saved_path}")


if __name__ == "__main__":
    main()